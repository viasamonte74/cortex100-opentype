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

exec python -m opentype_miner.train.harness_sft \
  --data \
    data/public/chat/chat_ops.jsonl \
    data/public/chat/chat_sql.jsonl \
    data/public/chat/chat_paint_spec.jsonl \
  --model "${CHAMPION_REPO}" \
  --revision "${CHAMPION_REV}" \
  --out adapters/harness_lora \
  --config configs/harness_lora.yaml \
  --device-map auto \
  --max-steps "${HARNESS_MAX_STEPS:-2000}"
