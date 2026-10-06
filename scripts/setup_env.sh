#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
# Build the SensorTSLM Python environment.
#
#   ./scripts/setup_env.sh [VENV_PATH] [cpu|cu126|cu129|cu130|cu132]
#
# The wheel bundles its own CUDA runtime, cuDNN and NCCL — the node needs an
# NVIDIA driver, not a CUDA toolkit. Pick the tag your driver supports:
#
#   cu126  driver >= 525 (CUDA 12.x)     cu130  driver >= 580 (CUDA 13.0)
#   cu129  driver >= 525 (CUDA 12.x)     cu132  driver >= 580 (CUDA 13.2)
#
# `nvidia-smi` reports the highest CUDA version the installed driver supports.
#
# cu128 is deliberately not offered: it has no torch 2.13.0 build (it stops at
# 2.11.0), so selecting it silently downgrades torch.
#
# Why this isn't just `pip install -r requirements.txt`:
#   opentslm depends on open-flamingo, which pins torch==2.0.1. Installing it
#   with its dependency graph downgrades torch and breaks CUDA support, so
#   opentslm/open-flamingo/einops-exts go in with --no-deps and the pieces
#   open_flamingo actually imports (open-clip-torch) are installed separately.
set -euo pipefail

VENV="${1:-$(dirname "$(dirname "$(realpath "$0")")")/../.venv}"
FLAVOR="${2:-cu126}"

case "$FLAVOR" in
  cpu|cu126|cu129|cu130|cu132) ;;
  *) echo "Unsupported flavor: $FLAVOR (want cpu|cu126|cu129|cu130|cu132)" >&2; exit 1 ;;
esac

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found — installing to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Creating venv at $VENV (flavor: $FLAVOR)"
uv venv --python 3.12 "$VENV"
PY="$VENV/bin/python"

TORCH_INDEX="https://download.pytorch.org/whl/$FLAVOR"

# Pin torch so a variant without the expected build fails loudly here rather
# than resolving to an older one deep in the dependency graph.
TORCH_SPEC="torch==2.13.0"

echo "==> Installing $TORCH_SPEC ($FLAVOR)"
uv pip install --python "$PY" "$TORCH_SPEC" --index-url "$TORCH_INDEX"

echo "==> Installing ML stack"
uv pip install --python "$PY" \
  --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match \
  transformers peft accelerate huggingface-hub numpy pandas pyarrow scipy \
  tqdm scikit-learn einops matplotlib datasets python-dotenv openai

# opentslm and open-flamingo carry pins that would clobber torch; their import
# paths only need open-clip-torch, which we install with its own deps.
echo "==> Installing opentslm + open-flamingo (--no-deps)"
uv pip install --python "$PY" --no-deps opentslm open-flamingo einops-exts

echo "==> Installing open-clip-torch (pulls timm/torchvision)"
uv pip install --python "$PY" \
  --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match \
  open-clip-torch

echo "==> Verifying"
"$PY" - <<'EOF'
import torch, transformers
print(f"torch        {torch.__version__}  cuda_available={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
print(f"transformers {transformers.__version__}")
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP  # noqa: F401
from opentslm.prompt.full_prompt import FullPrompt  # noqa: F401
print("opentslm     imports OK")
EOF

echo
echo "Done. Activate with: source $VENV/bin/activate"
