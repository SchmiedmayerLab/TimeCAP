#!/bin/bash
# Smoke test: text-only LLM backend (HF text-generation + Gruver tokenizer).
set -euo pipefail

set -a
source ~/SensorTSLM/.env
set +a
source ~/SensorTSLM/scripts/sherlock_setup.sh > /dev/null 2>&1
cd ~/SensorTSLM
export HF_HOME=$SCRATCH/hf_cache
export MHC_EXPORT_DIR=$SCRATCH/exports/lean_full_stride7
export MHC_LABELS_PATH=$SCRATCH/weekly_labels_lookup_ours_stride7.parquet
export MAX_SAMPLES=1

echo "=== host: $(hostname) ==="

python evaluation/run_eval.py \
  --task biological_sex \
  --backend text-local \
  --model meta-llama/Llama-3.2-1B
