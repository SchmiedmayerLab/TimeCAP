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
to the current directory: `metrics.json`, `predictions.csv`, and
`run_config.json` (the settings and data paths the run used — what tells
two runs of the same checkpoint apart).

`MAX_SAMPLES` above caps the run for a smoke test. It is read from the
environment, so an exported value silently truncates every later eval in
that shell; `scripts/gpu_sweep.py` clears it for exactly this reason.

In `metrics.json`, `n_samples` counts the samples that were *scored* —
predictions that failed to parse are excluded and counted in `n_unknown`.
`n_evaluated` is the number the model was run on, and is the denominator
a results table wants.

## Available tasks

- `biological_sex` — binary (Male / Female)
- `bmi` — ordinal (Underweight … Morbid Obesity)
- `diabetes` — binary
- `hypertension` — binary
- `cardiovascular` — binary

## WESAD Generalization Eval

Use `run_wesad_eval.py` to evaluate an MHC-trained OpenTSLM checkpoint on
WESAD state labels. The task is 4-way classification:
`baseline`, `stress`, `amusement`, `meditation`.

The default dataset windowing uses non-overlapping 120-second windows and keeps
only windows whose majority WESAD label has at least 0.8 purity. WESAD label
channels are removed from model inputs to avoid answer leakage.

```bash
python evaluation/run_wesad_eval.py \
  --model meta-llama/Llama-3.2-1B \
  --opentslm-checkpoint /path/to/mhc_trained_checkpoint.pt \
  --wesad-dataset-dir /path/to/wesad_timef \
  --window-seconds 120 \
  --min-label-purity 0.8 \
  --save-every 50
```

For a quick server smoke test, add `--max-samples 2`. Results are written to
`results/wesad_state/<checkpoint-name>/<timestamp>/`.

## MIMIC-IV Mortality Eval

Use `run_mimic_eval.py` to evaluate an MHC-trained OpenTSLM checkpoint on
MIMIC-IV in-hospital mortality using chart-events-only ICU time series. The
task is binary classification: `survived` or `died`.

The default dataset windowing uses the first 24 hours after ICU admission,
hourly median bins, and the selected chart-event channels HR, SBP, DBP, MAP,
RR, SpO2, temperature, and glucose. The mortality label comes from
`admissions.hospital_expire_flag` and is never included in model inputs. By
default, the cohort requires `icustays.outtime >= icustays.intime + 24h` so a
time series ending early does not reveal early death or early ICU discharge.

```bash
python evaluation/run_mimic_eval.py \
  --model meta-llama/Llama-3.2-1B \
  --opentslm-checkpoint /path/to/mhc_trained_checkpoint.pt \
  --mimic-data-dir /path/to/mimiciv \
  --observation-hours 24 \
  --bin-hours 1 \
  --min-observed-channels 1 \
  --require-full-observation-window \
  --save-every 50
```

For a quick server smoke test, add `--max-samples 2`. For an ablation that
keeps shorter ICU stays, pass `--no-require-full-observation-window`. Results
are written to `results/mimic_mortality/<checkpoint-name>/<timestamp>/`.
