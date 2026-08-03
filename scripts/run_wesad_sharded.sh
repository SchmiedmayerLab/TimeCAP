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
# Defaults to the joint reference checkpoint; the parallel axis is windows, not
# checkpoints. ARM=label_only switches to the metabolic-only run.
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

# Training arm to evaluate. "joint" is the caption+metabolic run; "label_only"
# is the metabolic-only run with no captioning objective — the comparison that
# says whether captions bought anything on an unseen dataset. CHECKPOINT
# overrides both.
#
# label_only points at metabolic_bs2_step66k.pt, which despite its name is at
# step 92,799 — byte-identical in step and loss to stage_metabolic/latest.pt,
# and ~2.9x more trained than best_metabolic.pt. It is used in preference to
# latest.pt because both stages contain a latest.pt and results are keyed on
# the checkpoint stem, so the two arms would collide in one results dir.
#
# The arms differ in step count (joint 227,999 vs 92,799) because the joint
# training set is the metabolic label QA plus captions, while label_only is the
# labels alone — the 2.46x ratio tracks the dataset size, and both are at epoch
# 0. Label exposure is therefore the same in both arms and the extra steps are
# the caption objective itself, which is what the ablation is asking about.
# Equalising steps would give the label-only arm extra epochs over the same
# labels and answer a different question.
ARM="${ARM:-joint}"
CKPT_ROOT="$REPO/results/meta-llama_Llama-3.2-1B/OpenTSLMSP"
case "$ARM" in
  joint)      DEFAULT_CKPT="$CKPT_ROOT/stage_joint_caption_metabolic/checkpoints/joint_bs2_final_1b.pt" ;;
  label_only) DEFAULT_CKPT="$CKPT_ROOT/stage_metabolic/checkpoints/metabolic_bs2_step66k.pt" ;;
  *) echo "ARM must be joint or label_only, got $ARM" >&2; exit 1 ;;
esac
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CKPT}"
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
