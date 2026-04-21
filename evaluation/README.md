# Evaluation Pipeline

LLM-based evaluation of MHC metabolic labels. For each weekly sample the
pipeline renders per-channel time-series plots, prepends the extracted
caption, asks a model to predict a target label, parses the answer and
computes classification metrics.

Currently Sherlock-specific — paths below assume `$SCRATCH` and the
MHC-benchmark repo at `~/MHC-benchmark`.

## Prerequisites

| Artifact | Path | How to build |
|---|---|---|
| Weekly HF dataset (stride=7) | `$SCRATCH/mhc-benchmark-weekly_hf_stride7/` | [1] |
| Captioned arrow shards | `$SCRATCH/exports/lean_full_stride7/shard_{0..3}/recordings_*.arrow` | [2] |
| Labels lookup parquet | `$SCRATCH/weekly_labels_lookup_ours_stride7.parquet` | [3] |
| OpenAI API key | `~/SensorTSLM/.env` (`OPENAI_API_KEY=...`) | manual |

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

The sharded caption script (`scripts/export_captions_sharded.sh`) lives
only on `main`, not on `feature/evaluation-pipeline`. Switch branches,
submit, switch back. Four parallel sbatch jobs, ~30 min total.

```bash
cd ~/SensorTSLM
git checkout main
export MHC_WEEKLY_DATASET_DIR=$SCRATCH/mhc-benchmark-weekly_hf_stride7
export TOTAL_ROWS=156428
./scripts/export_captions_sharded.sh weekly 4 $SCRATCH/exports/lean_full_stride7
# wait: squeue -u $USER
git checkout feature/evaluation-pipeline
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

```bash
srun --mem=16G -c 2 -t 00:15:00 -p normal bash -c '
  ml python/3.12.1 && ml libsndfile && \
  source $SCRATCH/sensor-venv/bin/activate && \
  export HF_HOME=$SCRATCH/hf_cache && \
  cd ~/SensorTSLM && \
  python evaluation/run_eval.py \
    --dataset mhc_metabolic --task biological_sex --model gpt-4o \
    --export-dir $SCRATCH/exports/lean_full_stride7 \
    --labels-path $SCRATCH/weekly_labels_lookup_ours_stride7.parquet \
    --max-samples 5 --split test
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
