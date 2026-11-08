#!/usr/bin/env bash
# One-time setup on the GPU VM (Oracle Linux 9 GPU image, user opc; NVIDIA driver preinstalled).
#   ssh opc@<ip> 'bash -s' < ops/remote_setup.sh
set -euo pipefail

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
sudo dnf install -y -q git tmux >/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

[ -d whisper-t5-bridge ] || git clone -q https://github.com/bryceadam1/whisper-t5-bridge.git
cd whisper-t5-bridge && git pull -q
uv python install -q 3.12
[ -d .venv ] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt

# Pre-fetch models; the TTS model at its pinned revision (its custom code was reviewed before use).
.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
from scripts.tts_generate import MODEL_ID, MODEL_REVISION
snapshot_download(MODEL_ID, revision=MODEL_REVISION)
for m in ("openai/whisper-base.en", "google/flan-t5-small"):
    snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
EOF
df -h ~ | tail -1
