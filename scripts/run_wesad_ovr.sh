#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
#
# SPDX-License-Identifier: MIT
#
# One-vs-rest WESAD evaluation across local GPUs, then merge and combine.
#
#   ./scripts/run_wesad_ovr.sh                  # all GPUs, reference checkpoint
#   NUM_GPUS=4 ./scripts/run_wesad_ovr.sh
#   ./scripts/run_wesad_ovr.sh --max-samples 8  # smoke test
#
# Four states x (NUM_GPUS/4) window shards = NUM_GPUS jobs, one per GPU. With 8
# GPUs that is each state evaluated over two shards. Only the reference
# checkpoint (joint_bs2_final_1b) is used: the caption families are an
# MHC-training ablation and say nothing about WESAD transfer.
#
# Runs to completion then merges shards per state and combines the four into a
# 4-way result. Safe to re-run; each launch writes its own timestamped dirs.
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
STATES=(baseline stress amusement meditation)

[ -f "$CHECKPOINT" ] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
[ -d "$WESAD_DIR" ] || { echo "Missing WESAD dataset: $WESAD_DIR (run scripts/build_wesad_timef.py)" >&2; exit 1; }
[ -x "$VENV/bin/python" ] || { echo "No venv at $VENV — run scripts/setup_env.sh" >&2; exit 1; }

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS="$("$VENV/bin/python" -c 'import torch; print(torch.cuda.device_count())')"
fi
[ "$NUM_GPUS" -ge 1 ] || { echo "No GPUs visible" >&2; exit 1; }

# Shards per state. With fewer than 4 GPUs the states queue up on what's there
# rather than oversubscribing a card.
SHARDS=$(( NUM_GPUS / ${#STATES[@]} ))
[ "$SHARDS" -ge 1 ] || SHARDS=1

# run_wesad_eval writes ./results relative to its cwd, so anchor to the repo.
cd "$REPO"
mkdir -p wesad_ovr_logs

echo "GPUs: $NUM_GPUS | states: ${#STATES[@]} | shards/state: $SHARDS | batch: $BATCH_SIZE"
echo

gpu=0
pids=()
labels=()
for state in "${STATES[@]}"; do
  for ((s = 0; s < SHARDS; s++)); do
    dev=$(( gpu % NUM_GPUS ))
    args=(--model "$MODEL"
          --opentslm-checkpoint "$CHECKPOINT"
          --wesad-dataset-dir "$WESAD_DIR"
          --binary-label "$state"
          --batch-size "$BATCH_SIZE"
          --max-new-tokens "$MAX_NEW_TOKENS")
    # Only pass sharding flags when actually sharding: --num-shards 1 skips the
    # shard suffix, and merge_eval_shards would then find nothing to merge.
    [ "$SHARDS" -gt 1 ] && args+=(--num-shards "$SHARDS" --shard-index "$s")

    log="wesad_ovr_logs/${state}_shard${s}.log"
    CUDA_VISIBLE_DEVICES="$dev" "$VENV/bin/python" evaluation/run_wesad_eval.py \
      "${args[@]}" "$@" > "$log" 2>&1 &
    pids+=($!)
    labels+=("$state shard $s (GPU $dev)")
    echo "  launched ${labels[-1]} -> $log"
    gpu=$(( gpu + 1 ))
  done
done

echo
status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "FAILED: ${labels[$i]}" >&2
    status=1
  fi
done
[ "$status" -eq 0 ] || { echo "See wesad_ovr_logs/ for details." >&2; exit 1; }

echo "All jobs finished."
if [ "$SHARDS" -gt 1 ]; then
  echo
  for state in "${STATES[@]}"; do
    "$VENV/bin/python" scripts/merge_eval_shards.py --task "wesad_${state}"
  done
fi

echo
"$VENV/bin/python" scripts/combine_wesad_ovr.py
