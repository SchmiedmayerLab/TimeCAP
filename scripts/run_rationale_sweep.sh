#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
# One train_rationale.py job (SFT -> baseline eval -> GRPO -> test eval) per GPU, one seed each.
# Seeds beyond the GPU count queue behind the first free GPU.
#
#   VENV=~/venv HF_HOME=~/hf_cache ./scripts/run_rationale_sweep.sh --gpus 8 --seeds "0 1 2 3 4 5 6 7"
#   ... --dry-run                      print the jobs only
#   ... -- --limit 8 --max_new_tokens 64   anything after -- goes to train_rationale.py
#
# Re-running the same command resumes: finished seeds (curriculum_results.json) are skipped,
# finished stages inside a seed are skipped, and GRPO continues from latest.pt.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HOME/venv}"
GPUS=8
SEEDS="0 1 2 3 4 5 6 7"
OUT="$ROOT/runs/rationale"
SFT_FILE="$ROOT/rationale_sft_v4.arrow"
GRPO_FILE="$ROOT/rationale_grpo_v1.arrow"
DRY=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

for f in "$SFT_FILE" "$GRPO_FILE" "$VENV/bin/python"; do
  [[ -e "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done
[[ -n "${HF_HOME:-}" ]] || { echo "set HF_HOME (local copy of the HF cache with Llama-3.2-1B)" >&2; exit 1; }
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false

mkdir -p "$ROOT/train_logs"
git -C "$ROOT" rev-parse --short HEAD 2>/dev/null > "$OUT.revision" || cat "$ROOT/REVISION" > "$OUT.revision" 2>/dev/null || true

todo=()
for seed in $SEEDS; do
  if compgen -G "$OUT/seed$seed/*/*/curriculum_results.json" >/dev/null; then
    echo "seed $seed: complete, skipping"
  else
    todo+=("$seed")
  fi
done
echo "${#todo[@]} seed(s) to run on $GPUS GPU(s): ${todo[*]:-none}"
echo "  sft : $SFT_FILE"
echo "  grpo: $GRPO_FILE"
echo "  out : $OUT/seed<N>   logs: $ROOT/train_logs/rationale_seed<N>.log"
[[ $DRY -eq 1 || ${#todo[@]} -eq 0 ]] && exit 0

declare -A gpu_of
running=()
for seed in "${todo[@]}"; do
  # wait for a free GPU slot
  while [[ ${#running[@]} -ge $GPUS ]]; do
    wait -n || true
    alive=()
    for pid in "${running[@]}"; do kill -0 "$pid" 2>/dev/null && alive+=("$pid"); done
    running=("${alive[@]}")
  done
  used=" "
  for pid in "${running[@]}"; do used+="${gpu_of[$pid]} "; done
  for ((g = 0; g < GPUS; g++)); do [[ "$used" != *" $g "* ]] && break; done

  log="$ROOT/train_logs/rationale_seed$seed.log"
  echo "seed $seed -> GPU $g ($log)"
  (cd "$ROOT" && CUDA_VISIBLE_DEVICES=$g exec "$VENV/bin/python" train_rationale.py \
      --seed "$seed" --results_dir "$OUT/seed$seed" \
      --sft_file "$SFT_FILE" --grpo_file "$GRPO_FILE" --resume "${EXTRA[@]}") >>"$log" 2>&1 &
  gpu_of[$!]=$g
  running+=("$!")
done
wait
echo "all seeds finished; per-seed results: $OUT/seed<N>/*/OpenTSLMSP/rationale_*/results/"
