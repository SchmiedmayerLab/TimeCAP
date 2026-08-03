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
#   POSITIVE_CLASS=survived ./scripts/run_mimic_sharded.sh # "discharged alive?"
#   ARM=label_only ./scripts/run_mimic_sharded.sh         # metabolic-only ckpt
#   ./scripts/run_mimic_sharded.sh --max-samples 8        # smoke test
#
# Defaults to LABEL_SPACE=truefalse. An MHC-trained checkpoint answers
# True/False for label spaces it never trained on, so asking for survived/died
# yields parse_rate ~0 — run native only to quantify that gap, not to get a
# mortality number.
#
# Defaults to the joint reference checkpoint. The caption-family ablation
# checkpoints are an MHC-training ablation and say nothing about transfer, so
# they are not swept here — pass CHECKPOINT explicitly if you want one.
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
# Still not compute-matched: joint is at step 227,999 vs 92,799 here, so a
# joint win is confounded by training length. joint_bs2_step167k.pt narrows it
# but does not close it. Pass CHECKPOINT to compare best-vs-best instead
# (best_model.pt / best_metabolic.pt), which trades compute matching for
# validation-selected weights.
ARM="${ARM:-joint}"
CKPT_ROOT="$REPO/results/meta-llama_Llama-3.2-1B/OpenTSLMSP"
case "$ARM" in
  joint)      DEFAULT_CKPT="$CKPT_ROOT/stage_joint_caption_metabolic/checkpoints/joint_bs2_final_1b.pt" ;;
  label_only) DEFAULT_CKPT="$CKPT_ROOT/stage_metabolic/checkpoints/metabolic_bs2_step66k.pt" ;;
  *) echo "ARM must be joint or label_only, got $ARM" >&2; exit 1 ;;
esac
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CKPT}"
MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"
MIMIC_DIR="${MIMIC_DIR:-$REPO/data/mimic-iv-clinical-database-demo-2.2}"
LABEL_SPACE="${LABEL_SPACE:-truefalse}"
# Which outcome True names. Run both and compare: a True-biased model
# scores near the majority rate one way and near zero the other.
POSITIVE_CLASS="${POSITIVE_CLASS:-died}"
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

echo "GPUs: $NUM_GPUS | label space: $LABEL_SPACE (True=$POSITIVE_CLASS) | batch: $BATCH_SIZE"
echo "Each shard re-parses chartevents independently — expect a few minutes of"
echo "dataset build per shard before generation starts."
echo

# One stamp for the whole launch. Shards cannot each take their own: the stamp
# is applied after a dataset build whose duration varies per shard, so they land
# in different seconds and merge_eval_shards sees N incomplete sets of 1 rather
# than one set of N.
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

pids=()
labels=()
for ((i = 0; i < NUM_GPUS; i++)); do
  args=(--model "$MODEL"
        --opentslm-checkpoint "$CHECKPOINT"
        --mimic-data-dir "$MIMIC_DIR"
        --label-space "$LABEL_SPACE"
        --positive-class "$POSITIVE_CLASS"
        --batch-size "$BATCH_SIZE"
        --max-new-tokens "$MAX_NEW_TOKENS"
        --run-stamp "$RUN_STAMP")
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
  if [ "$LABEL_SPACE" = "truefalse" ]; then
    task="mimic_mortality_truefalse"
    [ "$POSITIVE_CLASS" = "survived" ] && task="mimic_mortality_truefalse_survived"
  fi
  echo
  # Results are keyed on the checkpoint stem, so a task dir holding both arms
  # has two model subdirs and merge cannot pick one on its own.
  ckpt_stem="$(basename "$CHECKPOINT" .pt)"
  "$VENV/bin/python" scripts/merge_eval_shards.py \
    --task "$task" --model "opentslm-$ckpt_stem"
fi
