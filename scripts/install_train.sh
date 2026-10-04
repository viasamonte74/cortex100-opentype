#!/usr/bin/env bash
# Two-step install: CUDA torch from PyTorch's index, then train extras from PyPI.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"

echo "1/2 torch from ${CUDA_INDEX}"
uv pip install "torch>=2.6" --index-url "${CUDA_INDEX}"

echo "2/2 transformers/peft/accelerate from PyPI (torch already satisfied)"
uv pip install -e ".[train]"

python - <<'PY'
import torch, transformers, peft, accelerate
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("accelerate", accelerate.__version__)
PY
