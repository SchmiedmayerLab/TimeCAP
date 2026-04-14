#!/bin/bash
# Submit sharded caption export jobs on Sherlock.
# Usage: ./scripts/export_captions_sharded.sh [NUM_SHARDS] [OUT_DIR]
#   NUM_SHARDS  default 4
#   OUT_DIR     default $SCRATCH/exports/lean_full

set -euo pipefail

NUM_SHARDS="${1:-4}"
OUT_DIR="${2:-$SCRATCH/exports/lean_full}"
TOTAL_ROWS="${TOTAL_ROWS:-615490}"
WEEKLY_DIR="${MHC_WEEKLY_DATASET_DIR:-/scratch/users/mrosenbl/data/processed/weekly_hf/}"
REPO_DIR="${REPO_DIR:-$HOME/SensorTSLM}"

mkdir -p "$REPO_DIR/logs"
STRIDE=$(( (TOTAL_ROWS + NUM_SHARDS - 1) / NUM_SHARDS ))

echo "Submitting $NUM_SHARDS shards of ~$STRIDE rows each"
echo "  out:    $OUT_DIR/shard_<i>"
echo "  source: $WEEKLY_DIR"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  START=$(( i * STRIDE ))
  END=$(( START + STRIDE ))
  sbatch -p normal -c 2 --mem=8G -t 02:00:00 \
    -o "$REPO_DIR/logs/caption-$i-%j.out" \
    -J "caption-$i" \
    --wrap="ml python/3.12.1; ml libsndfile; \
source \$SCRATCH/sensor-venv/bin/activate; \
export MHC_WEEKLY_DATASET_DIR=$WEEKLY_DIR; \
cd $REPO_DIR; \
python scripts/export_captions.py --out $OUT_DIR/shard_$i --start $START --end $END"
done

echo "Check: squeue -u \$USER"
