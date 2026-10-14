<!--

This source file is part of the TimeCAP open-source project

SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT

-->

# Evaluation Pipeline

LLM-based evaluation of MHC metabolic labels. For each weekly sample the
pipeline hands per-channel signals to a model (as plots or an encoded
tensor depending on the backend), asks it to predict a target label,
parses the answer and computes classification metrics.

Currently Sherlock-specific — paths below assume `$SCRATCH` and the
MHC-benchmark repo at `~/MHC-benchmark`.

## Prerequisites

| Artifact | Path | How to build |
|---|---|---|
| Weekly HF dataset (stride=7) | `$SCRATCH/mhc-benchmark-weekly_hf_stride7/` | [1] |
| Captioned arrow shards | `$SCRATCH/exports/lean_full_stride7/shard_{0..3}/recordings_*.arrow` | [2] |
| Labels lookup parquet | `$SCRATCH/weekly_labels_lookup_ours_stride7.parquet` | [3] |
| OpenAI API key | `~/TimeCap/.env` (`OPENAI_API_KEY=...`) | manual |

Upstream MHC-benchmark deps (one-time DVC pulls inside `~/MHC-benchmark`,
using its conda env):

```bash
cd ~/MHC-benchmark
dvc pull data/processed/daily_hourly_hf.dvc   # ~4 GB, feeds [1]
dvc pull data/labels.dvc                       # ~600 MB, feeds [3]
```

## [1] Weekly HF dataset (stride=7)

Non-overlapping 7-day windows aligned to each user's first valid day.
~156k rows, ~40 min.

```bash
srun --mem=32G -c 4 -t 02:30:00 -p normal bash -c '
  export PATH=$SCRATCH/mhc-benchmark/miniforge3/envs/mhc-benchmark-run/bin:$PATH
  export PYTHONPATH=$HOME/MHC-benchmark/src:$HOME/MHC-benchmark
  cd ~/MHC-benchmark
  python -u -m data.processing.daily_hourly_hf_to_weekly_hf \
    --daily-hourly-path data/processed/daily_hourly_hf \
    --output-path $SCRATCH/mhc-benchmark-weekly_hf_stride7 \
    --stride-days 7
'
```

## [2] Captioned arrow shards

Four parallel sbatch jobs, ~30 min total.

```bash
cd ~/TimeCap
export MHC_WEEKLY_DATASET_DIR=$SCRATCH/mhc-benchmark-weekly_hf_stride7
export TOTAL_ROWS=156428
./scripts/export_captions_sharded.sh weekly 4 $SCRATCH/exports/lean_full_stride7
# wait: squeue -u $USER
```

## [3] Labels lookup parquet

Reads `(user_id, week_start)` pairs from the stride=7 HF dataset and
attaches labels from MHC-benchmark's `labels.json`. By construction
100% aligned to our captions. ~2 min.

```bash
srun --mem=16G -c 2 -t 00:30:00 -p normal bash -c '
  export PATH=$SCRATCH/mhc-benchmark/miniforge3/envs/mhc-benchmark-run/bin:$PATH
  cd ~/MHC-benchmark
  python scripts/build_labels_lookup.py \
    --hf-dir $SCRATCH/mhc-benchmark-weekly_hf_stride7 \
    --segment-type weekly \
    --output $SCRATCH/weekly_labels_lookup_ours_stride7.parquet
'
```

## Run Eval

Three backends, selected via `--backend` (default `openai`):

| Backend     | What `--model` means                  | Example |
|-------------|---------------------------------------|---------|
| `openai`    | Chat model name at `OPENAI_BASE_URL`  | `gpt-4o` (VLM, plots in prompt) |
| `vlm-local` | HF image-text-to-text model ID        | `google/gemma-3-4b-it` |
| `opentslm`  | HF Hub repo_id (auto-pulls backbone+weights) OR LLM backbone id with `--opentslm-checkpoint` for a local .pt | `OpenTSLM/llama-3.2-1b-har-sp` |

Configuration via env vars (all optional):
- `MHC_EXPORT_DIR` (default `exports`) — Arrow shard directory
- `MHC_LABELS_PATH` (default `data/labels_dev_subset.parquet`) — labels parquet
- `OPENAI_BASE_URL` (default `https://api.openai.com/v1`) — OpenAI-compatible endpoint
- `MAX_SAMPLES` (unset = full eval) — cap on samples for debug runs

Split is hardcoded to `test`.

```bash
srun --mem=16G -c 2 -t 00:15:00 -p normal bash -c '
  ml python/3.12.1 && ml libsndfile && \
  source $SCRATCH/sensor-venv/bin/activate && \
  export HF_HOME=$SCRATCH/hf_cache && \
  export MHC_EXPORT_DIR=$SCRATCH/exports/lean_full_stride7 && \
  export MHC_LABELS_PATH=$SCRATCH/weekly_labels_lookup_ours_stride7.parquet && \
  export MAX_SAMPLES=5 && \
  cd ~/TimeCap && \
  python evaluation/run_eval.py --task biological_sex --model gpt-4o
'
```

Results are written to `results/<label>/<model>/<timestamp>/` relative
to the current directory: `metrics.json` + `predictions.csv`.

## Available tasks

- `biological_sex` — binary (Male / Female)
- `bmi` — ordinal (Underweight … Morbid Obesity)
- `diabetes` — binary
- `hypertension` — binary
- `cardiovascular` — binary
