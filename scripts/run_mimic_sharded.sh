#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
#
# SPDX-License-Identifier: MIT
#
# MIMIC-IV mortality evaluation across local GPUs, sharded by ICU stay.
#
#   ./scripts/run_mimic_sharded.sh                        # truefalse labels
#   LABEL_SPACE=native ./scripts/run_mimic_sharded.sh     # survived/died
#   ./scripts/run_mimic_sharded.sh --max-samples 8        # smoke test
#
# Defaults to LABEL_SPACE=truefalse. An MHC-trained checkpoint answers
# True/False for label spaces it never trained on, so asking for survived/died
# yields parse_rate ~0 — run native only to quantify that gap, not to get a
# mortality number.
#
# Only the reference checkpoint (joint_bs2_final_1b) is used: the caption
# families are an MHC-training ablation and say nothing about MIMIC transfer.
#
# Each shard writes its own results dir; merge with scripts/merge_eval_shards.py
# (run automatically at the end). A single shard's metrics.json covers only its
# slice and is not the answer.
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
MIMIC_DIR="${MIMIC_DIR:-$REPO/data/mimic-iv-clinical-database-demo-2.2}"
LABEL_SPACE="${LABEL_SPACE:-truefalse}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

[ -f "$CHECKPOINT" ] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
[ -d "$MIMIC_DIR" ] || { echo "Missing MIMIC-IV data: $MIMIC_DIR" >&2; exit 1; }
[ -x "$VENV/bin/python" ] || { echo "No venv at $VENV — run scripts/setup_env.sh" >&2; exit 1; }

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS="$("$VENV/bin/python" -c 'import torch; print(torch.cuda.device_count())')"
fi
[ "$NUM_GPUS" -ge 1 ] || { echo "No GPUs visible" >&2; exit 1; }

# run_mimic_eval writes ./results relative to its cwd, so anchor to the repo.
cd "$REPO"
mkdir -p mimic_shard_logs

echo "GPUs: $NUM_GPUS | label space: $LABEL_SPACE | batch: $BATCH_SIZE"
echo "Each shard re-parses chartevents independently — expect a few minutes of"
echo "dataset build per shard before generation starts."
echo

pids=()
labels=()
for ((i = 0; i < NUM_GPUS; i++)); do
  args=(--model "$MODEL"
        --opentslm-checkpoint "$CHECKPOINT"
        --mimic-data-dir "$MIMIC_DIR"
        --label-space "$LABEL_SPACE"
        --batch-size "$BATCH_SIZE"
        --max-new-tokens "$MAX_NEW_TOKENS")
  # Only pass sharding flags when actually sharding: --num-shards 1 skips the
  # shard suffix, and merge_eval_shards would then find nothing to merge.
  [ "$NUM_GPUS" -gt 1 ] && args+=(--num-shards "$NUM_GPUS" --shard-index "$i")

  log="mimic_shard_logs/shard${i}.log"
  CUDA_VISIBLE_DEVICES="$i" "$VENV/bin/python" evaluation/run_mimic_eval.py \
    "${args[@]}" "$@" > "$log" 2>&1 &
  pids+=($!)
  labels+=("shard $i (GPU $i)")
  echo "  launched ${labels[-1]} -> $log"
done

echo
status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "FAILED: ${labels[$i]}" >&2
    status=1
  fi
done
[ "$status" -eq 0 ] || { echo "See mimic_shard_logs/ for details." >&2; exit 1; }

echo "All shards finished."
if [ "$NUM_GPUS" -gt 1 ]; then
  task="mimic_mortality"
  [ "$LABEL_SPACE" = "truefalse" ] && task="mimic_mortality_truefalse"
  echo
  "$VENV/bin/python" scripts/merge_eval_shards.py --task "$task"
fi
