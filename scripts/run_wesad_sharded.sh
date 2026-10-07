#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
#
# SPDX-License-Identifier: MIT
#
# Evaluate WESAD across local GPUs by splitting the windows, one shard per GPU.
#
#   ./scripts/run_wesad_sharded.sh [extra run_wesad_eval.py args...]
#
# Only the reference checkpoint (joint_bs2_final_1b) is evaluated — the caption
# families are an MHC-training ablation and say nothing about WESAD transfer, so
# the parallel axis here is windows, not checkpoints.
#
# Each shard writes its own results dir; merge them with
# scripts/merge_eval_shards.py, which recomputes metrics over the union. A
# single shard's metrics.json covers only its slice and is not the answer.
set -euo pipefail

REPO="$(dirname "$(dirname "$(realpath "$0")")")"
WORKSPACE="$(dirname "$REPO")"
VENV="${VENV:-$WORKSPACE/.venv}"

export HF_HOME="${HF_HOME:-$WORKSPACE/hf_cache}"
# Llama-3.2-1B is gated and already cached; stay offline rather than depend on
# the token still being valid.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

CHECKPOINT="${CHECKPOINT:-$REPO/results/meta-llama_Llama-3.2-1B/OpenTSLMSP/stage_joint_caption_metabolic/checkpoints/joint_bs2_final_1b.pt}"
MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"
WESAD_DIR="${WESAD_DIR:-$REPO/data/wesad_timef}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS="$("$VENV/bin/python" -c 'import torch; print(torch.cuda.device_count())')"
fi
[ "$NUM_GPUS" -ge 1 ] || { echo "No GPUs visible" >&2; exit 1; }

[ -f "$CHECKPOINT" ] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
[ -d "$WESAD_DIR" ] || { echo "Missing WESAD dataset: $WESAD_DIR (run scripts/build_wesad_timef.py)" >&2; exit 1; }
[ -x "$VENV/bin/python" ] || { echo "No venv at $VENV — run scripts/setup_env.sh" >&2; exit 1; }

# run_wesad_eval writes ./results relative to its cwd, so anchor to the repo.
cd "$REPO"
mkdir -p wesad_shard_logs

echo "Launching $NUM_GPUS shards over $WESAD_DIR"
pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES="$i" "$VENV/bin/python" evaluation/run_wesad_eval.py \
    --model "$MODEL" \
    --opentslm-checkpoint "$CHECKPOINT" \
    --wesad-dataset-dir "$WESAD_DIR" \
    --num-shards "$NUM_GPUS" --shard-index "$i" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    "$@" > "wesad_shard_logs/shard${i}.log" 2>&1 &
  pids+=($!)
  echo "  shard $i -> GPU $i (pid ${pids[-1]})"
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

if [ "$status" -ne 0 ]; then
  echo "At least one shard failed — see wesad_shard_logs/" >&2
  exit 1
fi

echo
echo "All shards done. Merge with:"
echo "  $VENV/bin/python scripts/merge_eval_shards.py"
