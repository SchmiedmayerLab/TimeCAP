#!/bin/bash
#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
# Smoke test: OpenAI VLM backend (plots + text via chat completions).
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
  --backend openai \
  --model gpt-4o
