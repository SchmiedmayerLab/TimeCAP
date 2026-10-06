#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
# Launch the caption-family ablation eval grid across all local GPUs.
#
#   ./scripts/run_ablation_sweep.sh [extra gpu_sweep.py args...]
#
# 4 caption-family checkpoints x 10 tasks, full test sets. Resumable: re-run
# the same command after an interruption and completed jobs are skipped.
#
# blood_pressure and sleep_time are excluded. On those two the model answers
# from a label space it was trained on ("True"/"False") rather than the one
# asked for, so parse rate sits at 0.000-0.036 for the 1B checkpoints and the
# surviving samples are selected by vocabulary overlap rather than by skill.
# They need a different eval protocol (forced choice), not more compute.
set -euo pipefail

REPO="$(dirname "$(dirname "$(realpath "$0")")")"
WORKSPACE="$(dirname "$REPO")"
VENV="${VENV:-$WORKSPACE/.venv}"

export HF_HOME="${HF_HOME:-$WORKSPACE/hf_cache}"
export MHC_SPLIT_FILE="${MHC_SPLIT_FILE:-$WORKSPACE/data/sharable_users_seed42_2026.json}"
export MHC_EXPORT_DIR="${MHC_EXPORT_DIR:-$WORKSPACE/data/sharable_weekly}"
export MHC_LABELS_PATH="${MHC_LABELS_PATH:-$WORKSPACE/data/weekly_labels_lookup_user.parquet}"
# Llama-3.2-1B is gated; it is already in hf_cache, so stay offline rather than
# depending on the token still being valid.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$REPO/results/meta-llama_Llama-3.2-1B/OpenTSLMSP/stage_joint_caption_metabolic/checkpoints}"

TASKS=(
  afib bmi biological_sex cad cardiovascular
  diabetes happiness hypertension sleep_diagnosis work
)

for path in "$MHC_SPLIT_FILE" "$MHC_LABELS_PATH"; do
  [ -e "$path" ] || { echo "Missing: $path" >&2; exit 1; }
done
[ -d "$MHC_EXPORT_DIR" ] || { echo "Missing export dir: $MHC_EXPORT_DIR" >&2; exit 1; }
[ -x "$VENV/bin/python" ] || { echo "No venv at $VENV — run scripts/setup_env.sh" >&2; exit 1; }

# run_eval writes to ./results relative to its working directory, and
# gpu_sweep's resume check reads the same path, so both must run from the repo.
cd "$REPO"

exec "$VENV/bin/python" scripts/gpu_sweep.py \
  --checkpoints-dir "$CHECKPOINTS_DIR" \
  --tasks "${TASKS[@]}" \
  "$@"
