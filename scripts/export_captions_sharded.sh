#!/bin/bash
#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
#
# SPDX-License-Identifier: MIT
#
# Submit sharded caption export jobs on Sherlock.
#
# Prerequisites:
#   source scripts/sherlock_setup.sh default   # modules + venv + HF_HOME
#   export MHC_WEEKLY_DATASET_DIR=<path>       # for weekly (or MHC_DATASET_DIR for daily)
#   export HF_TOKEN=<token>                    # if the HF dataset is gated
#
# Usage: ./scripts/export_captions_sharded.sh [VARIANT] [NUM_SHARDS] [OUT_DIR]
#   VARIANT     "weekly" (default) or "daily"
#   NUM_SHARDS  default 4
#   OUT_DIR     default $SCRATCH/exports/lean_full (weekly) or lean_daily (daily)
#
# Optional env overrides:
#   REPO_DIR    default $HOME/TimeCap
#   VENV_DIR    default $SCRATCH/sensor-venv  (used via sherlock_setup.sh)
#   TOTAL_ROWS  auto-detected for daily; defaults to 615490 for weekly
#   SPLIT_FILE  path to MHC-benchmark sharable_users_seed42_2026.json; when
#               set, non-sharable users are dropped at load time

set -euo pipefail

VARIANT="${1:-weekly}"
NUM_SHARDS="${2:-4}"
REPO_DIR="${REPO_DIR:-$HOME/TimeCap}"

if [ "$VARIANT" = "daily" ]; then
  DATA_DIR="${MHC_DATASET_DIR:-}"
  OUT_DIR="${3:-$SCRATCH/exports/lean_daily}"
  ENV_VAR="MHC_DATASET_DIR"
  # Auto-detect row count
  TOTAL_ROWS="${TOTAL_ROWS:-0}"
else
  DATA_DIR="${MHC_WEEKLY_DATASET_DIR:-}"
  OUT_DIR="${3:-$SCRATCH/exports/lean_full}"
  ENV_VAR="MHC_WEEKLY_DATASET_DIR"
  TOTAL_ROWS="${TOTAL_ROWS:-615490}"
fi

# Dataset path is per-user and must be set explicitly.
if [ -z "$DATA_DIR" ]; then
  echo "Error: $ENV_VAR is not set. Export $ENV_VAR=<path> before running." >&2
  exit 1
fi

# Auto-detect total rows if not set
if [ "$TOTAL_ROWS" -eq 0 ]; then
  echo "Detecting dataset size..."
  TOTAL_ROWS=$(python -c "from datasets import load_from_disk; print(len(load_from_disk('$DATA_DIR')))")
  echo "Detected $TOTAL_ROWS rows"
fi

mkdir -p "$REPO_DIR/logs"
STRIDE=$(( (TOTAL_ROWS + NUM_SHARDS - 1) / NUM_SHARDS ))

SPLIT_ARG=""
if [ -n "${SPLIT_FILE:-}" ]; then
  SPLIT_ARG="--split_file $SPLIT_FILE"
  echo "  split_file: $SPLIT_FILE"
fi

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
python scripts/export_captions.py --variant $VARIANT --out $OUT_DIR/shard_$i --start $START --end $END $SPLIT_ARG"
done

echo "Check: squeue -u \$USER"
