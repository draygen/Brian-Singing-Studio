#!/usr/bin/env bash
# Brian-Singing-Studio Phase 1 setup -- SoulX-Singer (SVS) on RTX 5070 / WSL.
#
# Creates engine/.venv (python 3.10), installs torch 2.8.0+cu128 (repo pins
# torch 2.2.0 which predates Blackwell sm_120 -- it cannot run on this GPU),
# installs the trimmed requirements, then downloads the SVS model and the
# preprocessing models needed for an ENGLISH prompt annotation only:
#   rmvpe (F0), rosvot (note transcription), parakeet-tdt-0.6b-v2 (EN ASR).
# Skipped deliberately: model-svc.pt (fallback only), zh ASR, vocal-sep
# models (Brian's reference is a dry mono recording), gradio/wandb/torchcodec.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="$ROOT/engine/SoulX-Singer"
VENV="$ROOT/engine/.venv"

[[ -d "$ENGINE" ]] || git clone --depth 1 https://github.com/Soul-AILab/SoulX-Singer.git "$ENGINE"

[[ -x "$VENV/bin/python" ]] || uv venv --python 3.10 "$VENV"
PY="$VENV/bin/python"

uv pip install --python "$PY" torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Repo requirements minus: torch/torchaudio (pinned above), torchcodec (never
# imported), gradio (webui only), wandb (training only).
grep -vE "^(torch|torchaudio|torchcodec|gradio|wandb)" "$ENGINE/requirements.txt" > /tmp/soulx_reqs.txt
uv pip install --python "$PY" -r /tmp/soulx_reqs.txt
uv pip install --python "$PY" huggingface_hub && "$PY" -m nltk.downloader averaged_perceptron_tagger averaged_perceptron_tagger_eng cmudict punkt "nemo_toolkit[asr]==2.6.1"

mkdir -p "$ENGINE/pretrained_models"
"$PY" - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("Soul-AILab/SoulX-Singer",
                  local_dir="$ENGINE/pretrained_models/SoulX-Singer",
                  allow_patterns=["model.pt", "config.yaml", "README.md"])
snapshot_download("Soul-AILab/SoulX-Singer-Preprocess",
                  local_dir="$ENGINE/pretrained_models/SoulX-Singer-Preprocess",
                  allow_patterns=["rmvpe/*", "rosvot/*", "parakeet-tdt-0.6b-v2/*", "speech_seaco_paraformer*/*"])
EOF

"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo SETUP_DONE
