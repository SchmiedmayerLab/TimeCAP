#!/bin/bash
# Sherlock environment setup.
# MUST be sourced, not executed (venv activation needs to persist in caller shell):
#   source scripts/sherlock_setup.sh default
#   source scripts/sherlock_setup.sh full
#
# Modes:
#   default  every-session: module loads + activate existing venv + env vars
#   full     one-time: create venv on $SCRATCH and install pinned deps

MODE="${1:-default}"
VENV_DIR="${VENV_DIR:-$SCRATCH/sensor-venv}"

if [[ "$MODE" != "default" && "$MODE" != "full" ]]; then
    echo "usage: source scripts/sherlock_setup.sh [default|full]" >&2
    return 1 2>/dev/null || exit 1
fi

# --- modules (both modes) ---
ml python/3.12.1
ml libsndfile

# default auto-bootstraps to full when venv is missing
if [[ "$MODE" == "default" && ! -d "$VENV_DIR" ]]; then
    echo "venv missing at $VENV_DIR — bootstrapping full install"
    MODE=full
fi

# --- one-time full install ---
if [[ "$MODE" == "full" ]]; then
    if [[ -d "$VENV_DIR" ]]; then
        echo "venv already exists at $VENV_DIR — skipping creation"
    else
        python3 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"

    export PIP_CACHE_DIR="$SCRATCH/pip_cache"
    export HF_HOME="$SCRATCH/hf_cache"

    python3 -m pip install --upgrade pip wheel

    # Sherlock compute nodes are glibc 2.17 (CentOS 7) — pin to last versions with
    # manylinux2014 wheels and force binary-only so pip errors loudly instead of
    # silently building from source.
    pip install --only-binary=:all: \
        "numpy<2.1" "scipy<1.15" "pyarrow<18" "matplotlib<3.10" "Pillow"

    pip install --only-binary=:all: \
        "torch<2.5" --index-url https://download.pytorch.org/whl/cu121

    pip install --only-binary=:all: \
        datasets transformers accelerate openai python-dotenv opentslm

    echo "Full setup done. Venv at $VENV_DIR is active."
    return 0 2>/dev/null || exit 0
fi

# --- default every-session ---
source "$VENV_DIR/bin/activate"

export HF_HOME="$SCRATCH/hf_cache"
export MHC_WEEKLY_DATASET_DIR="${MHC_WEEKLY_DATASET_DIR:-/scratch/users/mrosenbl/data/processed/weekly_hf/}"
# HF_TOKEN is gated — set it yourself: export HF_TOKEN=<token>

echo "Sherlock env ready: $(python --version), venv=$VENV_DIR"
