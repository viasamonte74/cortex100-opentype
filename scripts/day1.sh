#!/usr/bin/env bash
# Day-1 pipeline smoke: recon → tiny public data → oracle chat → local sim.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

opentype-miner recon --fetch-structured-server
opentype-miner data generate --plan --smoke --out data/public
opentype-miner data oracle-chat --in data/public --out data/public/chat

# Concatenate a small mixed eval set
MIX=data/public/smoke_mix.jsonl
: > "$MIX"
for f in data/public/decisions_l3.jsonl data/public/ops.jsonl data/public/sql.jsonl data/public/paint_spec.jsonl; do
  [[ -f "$f" ]] && head -n 16 "$f" >> "$MIX"
done

opentype-miner eval sim \
  --cases "$MIX" \
  --champion-skill blur:1.3 \
  --challenger-skill blur:1.05 \
  --scores-out results/smoke_scores.jsonl \
  --json-out results/smoke_verdict.json

echo "Day-1 smoke OK. Next: point --model at the live champion and run train reads/harness."
