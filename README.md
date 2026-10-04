# opentype-miner

Fine-tuning stack for the Cortex **OpenType** challenge (`opentype-challenge` **v2.2.0**).
Miners improve `google/diffusiongemma-26B-A4B-it` (init from the **live champion**) and submit
merged BF16 weights with a byte-identical base `config.json`.

North star: beat the champion with a broad ~20% relative loss cut; both duel halves clear
LCB with margin. Optimize **simulator composite LCB**, not train loss.

## Layout

```text
RECON.md                 pins + live champion notes
data/mix.yaml            closed-loop mix + ship gate
data/{public,closed,synth,on_policy,private_proxy}/
vendor/structured_server.py   pinned sha (must match challenge pins)
src/opentype_miner/
  data/     generate, oracle→chat, bank ingest, hard-mine
  eval/     sim_duel (scoring.verdict), parity
  train/    read-slot soft CE, harness block-diffusion LoRA, WiSE-FT
  export/   merge LoRA → duel-legal BF16
configs/    read_lora.yaml, harness_lora.yaml, nemo_harness_lora.yaml
results/ablations.md
```

## Setup

```bash
cd /workspace/opentype-miner
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
# training extras — use the helper (torch from cu128, rest from PyPI):
bash scripts/install_train.sh
# equivalent:
#   uv pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cu128
#   uv pip install -e ".[train]"
opentype-miner recon --fetch-structured-server
```



## Day-1 smoke (no GPU weights)

```bash
bash scripts/day1.sh
```

## Training loop (48–72h cadence)

1. **RECON** — pin challenge `2.2.0` + `structured_server` sha; record champion.
2. **Sim** — base + champion headroom table (`opentype-miner eval sim`).
3. **Read-slot SFT** until forfeit ≈ 0 + **parity pass** vs vLLM+structured_server.
4. **Oracle harness SFT** (short JSON) from `data oracle-chat`.
5. Bank / sealed / prose aug; private-proxy holdout stays eval-only.
6. 2–3 rounds on-policy RFT/DPO + recovery splices; `data hard-mine`.
7. mix re-fit + WiSE-FT α sweep; log kills in `results/ablations.md`.
8. **Export** merged BF16 + base `config.json`; submit manually (no auto-submit).

### Reads (never plain TRL causal SFT)

```bash
# Init --model/--revision from the live champion
opentype-miner train reads train \
  --data data/public/decisions_l3.jsonl data/public/longctx_l2.jsonl \
  --model <champion_repo> --revision <champion_sha> \
  --out adapters/reads_lora --config configs/read_lora.yaml --max-steps 200
```

Hard priors: pinned `structured_server`, 1-step canvas denoise (256), soft posterior on
label tokens, `−log(label_mass)` penalty, never hard-label AR.

### Harness (block-diffusion CE on assistant JSON)

```bash
opentype-miner data oracle-chat --in data/public --out data/public/chat
opentype-miner train harness \
  --data data/public/chat/chat_ops.jsonl data/public/chat/chat_sql.jsonl \
  --model <champion_repo> --revision <champion_sha> \
  --out adapters/harness_lora --config configs/harness_lora.yaml --max-steps 200
```

If NeMo AutoModel is available (8×GPU, EP=8), use `configs/nemo_harness_lora.yaml`.

### Parity (test #1)

Serve the student with the same vLLM flags as the duel worker, front with pinned
`structured_server.py`, then:

```bash
opentype-miner eval parity \
  --cases data/public/decisions_l3.jsonl \
  --offline-url http://127.0.0.1:8011 \
  --served-url http://127.0.0.1:8012 \
  --max-cases 32
```

Slot logprobs must stay within BF16 of the served reader before any big run.

### Export + submit

```bash
opentype-miner export \
  --adapter adapters/reads_lora \
  --base <champion_or_merged_init> \
  --out exports/candidate \
  --hf-repo you/opentype-challenger
# push only *.safetensors (+ index) and config.json
opentype-challenge miner submit --api https://<master>/challenge/opentype \
  --repo you/opentype-challenger --revision <40hex> \
  --wallet-name w --wallet-hotkey h
```

## Hard rules (from challenge source)

| Rule | Source |
| --- | --- |
| Crown `g_LCB ≥ -ln 0.95` both halves | `scoring.G_MIN` |
| ≥2 tracks gain; no track regresses | `GAIN_TRACKS`, track guard |
| Weights 0.35 / 0.25 / 0.15 / 0.10 / 0.15 | `tracks.DEFAULT_PLAN` |
| Reads forfeit if `label_mass < 0.5` or off-label argmax | `decision_loss` |
| Export `config.json` byte-equal to base | `pins.BASE_FILES` / worker `assemble` |
| Adapters never ship | miner guide |

## Your decisions

GPU count/class, wallet/hotkey, paraphrase LLM + budget, HF repo name.
