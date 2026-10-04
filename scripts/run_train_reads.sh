#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export CHAMPION_REPO="${CHAMPION_REPO:-ebobo/m-0e98bd7f}"
export CHAMPION_REV="${CHAMPION_REV:-5824c0e82d99edb1b1aafc2ba424f564bd277c72}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 8bit does not fit (weights alone ~91GiB). Spend free VRAM on higher LoRA rank.
exec python -m opentype_miner.train.read_slot train \
  --data \
    data/public/decisions_l3.jsonl \
    data/public/decisions_l4.jsonl \
  --model "${CHAMPION_REPO}" \
  --revision "${CHAMPION_REV}" \
  --out adapters/reads_lora \
  --config configs/read_lora.yaml \
  --device-map auto \
  --quantize "${QUANTIZE:-4bit}" \
  --lora-r "${LORA_R:-64}" \
  --lora-alpha "${LORA_ALPHA:-128}" \
  --gradient-checkpointing \
  --max-prompt-len "${MAX_PROMPT_LEN:-2048}" \
  --max-steps "${READ_MAX_STEPS:-2000}"
