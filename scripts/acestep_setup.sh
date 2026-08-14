#!/usr/bin/env bash
# ACE-Step v1 (3.5B) in its own venv. torch 2.8.0+cu128 (Blackwell) instead of
# the README's cu126. Checkpoints auto-download to ~/.cache/ace-step on first
# run (~7 GB).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="$ROOT/engine/ACE-Step"
VENV="$ROOT/engine/.venv-ace"

[[ -d "$ENGINE" ]] || git clone --depth 1 https://github.com/ace-step/ACE-Step.git "$ENGINE"
[[ -x "$VENV/bin/python" ]] || uv venv --python 3.10 "$VENV"
uv pip install --python "$VENV/bin/python" torch==2.8.0 torchaudio==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$VENV/bin/python" -e "$ENGINE"
"$VENV/bin/python" -c "import torch, acestep; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo ACE_SETUP_DONE
