#!/usr/bin/env bash
# Quick environment bootstrap for an AWS notebook (or any fresh Linux box).
#
# Installs uv if missing, builds the project env from uv.lock, and verifies that
# torch sees the GPU. The lockfile already pins a CUDA-enabled torch on Linux, so
# `uv sync` alone is enough — no separate pytorch.org install needed.
#
# Usage (from a notebook cell, after cloning the repo):
#     !bash ai/scripts/bootstrap.sh
#
# Note: this runs in its own shell, so `uv` is only on PATH *inside* this script.
# To use `!uv ...` in later notebook cells, also run this once in a Python cell:
#     import os; os.environ["PATH"] = f"{os.path.expanduser('~')}/.local/bin:" + os.environ["PATH"]

set -euo pipefail

# Resolve the ai/ project dir from this script's location (ai/scripts/bootstrap.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_DIR="$(dirname "$SCRIPT_DIR")"

# 1. Ensure uv is installed and on PATH (the installer drops it in ~/.local/bin).
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo ">> Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo ">> uv $(uv --version)"

# 2. Build the environment from the lockfile.
cd "$AI_DIR"
echo ">> uv sync  (in $AI_DIR)"
uv sync

# 3. Verify torch / CUDA.
echo ">> Verifying torch + CUDA ..."
uv run python - <<'PY'
import torch
print(f"   torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   device: {torch.cuda.get_device_name(0)}")
else:
    print("   WARNING: no GPU visible to torch (check the instance's NVIDIA driver "
          "supports the locked CUDA build).")
PY

echo ""
echo ">> Done. Train with:"
echo "   cd $AI_DIR && uv run python -m oralskop.train.train --config configs/train/yolo11_seg.yaml"
