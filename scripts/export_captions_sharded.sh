#!/bin/bash
# Submit sharded caption export jobs on Sherlock.
# Usage: ./scripts/export_captions_sharded.sh [VARIANT] [NUM_SHARDS] [OUT_DIR]
#   VARIANT     "weekly" (default) or "daily"
#   NUM_SHARDS  default 4
#   OUT_DIR     default $SCRATCH/exports/lean_full (weekly) or lean_daily (daily)

set -euo pipefail

VARIANT="${1:-weekly}"
NUM_SHARDS="${2:-4}"
REPO_DIR="${REPO_DIR:-$HOME/SensorTSLM}"

if [ "$VARIANT" = "daily" ]; then
  DATA_DIR="${MHC_DATASET_DIR:-/scratch/users/mrosenbl/MHC-benchmark/data/processed/daily_hourly_hf}"
  OUT_DIR="${3:-$SCRATCH/exports/lean_daily}"
  ENV_VAR="MHC_DATASET_DIR"
  # Auto-detect row count
  TOTAL_ROWS="${TOTAL_ROWS:-0}"
else
  DATA_DIR="${MHC_WEEKLY_DATASET_DIR:-/scratch/users/mrosenbl/data/processed/weekly_hf/}"
  OUT_DIR="${3:-$SCRATCH/exports/lean_full}"
  ENV_VAR="MHC_WEEKLY_DATASET_DIR"
  TOTAL_ROWS="${TOTAL_ROWS:-615490}"
fi

# Auto-detect total rows if not set
if [ "$TOTAL_ROWS" -eq 0 ]; then
  echo "Detecting dataset size..."
  TOTAL_ROWS=$(python -c "from datasets import load_from_disk; print(len(load_from_disk('$DATA_DIR')))")
  echo "Detected $TOTAL_ROWS rows"
fi

mkdir -p "$REPO_DIR/logs"
STRIDE=$(( (TOTAL_ROWS + NUM_SHARDS - 1) / NUM_SHARDS ))

echo "Submitting $NUM_SHARDS shards of ~$STRIDE rows each ($VARIANT)"
echo "  out:    $OUT_DIR/shard_<i>"
echo "  source: $DATA_DIR"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  START=$(( i * STRIDE ))
  END=$(( START + STRIDE ))
  sbatch -p normal -c 2 --mem=8G -t 02:00:00 \
    -o "$REPO_DIR/logs/caption-${VARIANT}-$i-%j.out" \
    -J "caption-${VARIANT}-$i" \
    --wrap="ml python/3.12.1; ml libsndfile; \
source \$SCRATCH/sensor-venv/bin/activate; \
export ${ENV_VAR}=$DATA_DIR; \
cd $REPO_DIR; \
python scripts/export_captions.py --variant $VARIANT --out $OUT_DIR/shard_$i --start $START --end $END"
done

echo "Check: squeue -u \$USER"
