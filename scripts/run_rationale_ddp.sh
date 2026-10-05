#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
# Data-parallel GRPO: one model trained on all local GPUs, warm-started from an SFT checkpoint.
#
#   VENV=~/venv HF_HOME=~/hf_cache ./scripts/run_rationale_ddp.sh <sft best_model.pt> [--gpus 8] [-- extra args]
#
# Re-running the same command resumes from latest.pt.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HOME/venv}"
SFT_CKPT="${1:?usage: run_rationale_ddp.sh <sft best_model.pt> [--gpus N] [-- extra args]}"; shift
GPUS=8
OUT="$ROOT/runs/rationale_v2"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ -n "${HF_HOME:-}" ]] || { echo "set HF_HOME" >&2; exit 1; }
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$ROOT/train_logs" "$OUT"
cat "$ROOT/REVISION" > "$OUT/revision" 2>/dev/null || true

cd "$ROOT"
exec "$VENV/bin/torchrun" --standalone --nproc_per_node "$GPUS" train_rationale.py \
  --stages rationale_grpo --initial_checkpoint "$SFT_CKPT" --results_dir "$OUT" --resume \
  --grpo_file "$ROOT/rationale_grpo_v2.arrow" --no_baseline_eval \
  --prompts_per_generate 8 --grad_accum 8 --pg_chunk 2 --max_new_tokens 360 --eval_batch_size 32 \
  --grpo_lr_encoder 2e-5 --grpo_lr_projector 1e-5 --eval_every 100 --save_every_n_steps 50 \
  "${EXTRA[@]}" 2>&1 | tee -a "$ROOT/train_logs/rationale_ddp.log"
