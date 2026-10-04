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

# READ_DATA: space-separated jsonl paths (default l3+l4). Out defaults to adapters/reads_lora.
# Example level 5: READ_DATA=data/public/decisions_l5.jsonl READ_OUT=adapters/reads_lora_l5
# shellcheck disable=SC2206
DATA_FILES=(${READ_DATA:-data/public/decisions_l3.jsonl data/public/decisions_l4.jsonl})
OUT_DIR="${READ_OUT:-adapters/reads_lora}"

# 8bit does not fit (weights alone ~91GiB). Spend free VRAM on higher LoRA rank.
exec python -m opentype_miner.train.read_slot train \
  --data "${DATA_FILES[@]}" \
  --model "${CHAMPION_REPO}" \
  --revision "${CHAMPION_REV}" \
  --out "${OUT_DIR}" \
  --config configs/read_lora.yaml \
  --device-map auto \
  --quantize "${QUANTIZE:-4bit}" \
  --lora-r "${LORA_R:-64}" \
  --lora-alpha "${LORA_ALPHA:-128}" \
  --gradient-checkpointing \
  --max-prompt-len "${MAX_PROMPT_LEN:-2048}" \
  --max-steps "${READ_MAX_STEPS:-2000}"
