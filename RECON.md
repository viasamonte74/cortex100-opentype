# RECON — OpenType miner

Updated: 2026-10-04T01:31:23Z

## Pins (code is source of truth)

| Item | Value |
| --- | --- |
| challenge package | `2.2.0` (target **2.2.0**) |
| miner package | `0.1.0` |
| base | `google/diffusiongemma-26B-A4B-it@f7f5b7f5fa82ffc52addd066915886d497f5517b` |
| config.json sha256 | `13b11d2fe87302cc2332c64eb9eb4ac305d9b8a123ffe9c5cb5b1920fc70c506` |
| structured_server URL | `https://raw.githubusercontent.com/vllm-project/vllm/1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8/examples/features/structured_diffusion/structured_server.py` |
| structured_server sha256 | `7cd9aa0081090c064eaac28db0f54f812749eeb3ae787d7f7653d2e35d8a938f` |
| local structured_server | `/workspace/opentype-miner/vendor/structured_server.py` match=True actual=`7cd9aa0081090c064eaac28db0f54f812749eeb3ae787d7f7653d2e35d8a938f` |
| vLLM image | `vllm/vllm-openai:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423@sha256:ed3c505d2cf4b62b0ca8d0b87591bcb6fbff2a3abcc6d25732343e19b4d74f09` |
| canvas | `256` |
| max_model_len | `131072` |

## Live status

```json
{
  "champion": {
    "repo": "ebobo/m-0e98bd7f",
    "id": 11,
    "hotkey": "5Chndc77SXBhGA5BFVGQp5VrUPTSXTicLk3FunD2gKCnc9ny",
    "revision": "5824c0e82d99edb1b1aafc2ba424f564bd277c72",
    "digest": "dc87c2f62fe260a0bf9891c2e971b38edf248d1a9445b607a6f8ca6a3ce534e2",
    "g_lcb": 0.07038311281774265,
    "crowned_at": "2026-10-03T00:42:45Z"
  },
  "plan": {
    "decisions": {
      "weight": 0.35,
      "cases": 4000
    },
    "longctx": {
      "weight": 0.25,
      "cases": 800
    },
    "ops": {
      "weight": 0.15,
      "cases": 1000
    },
    "paint": {
      "weight": 0.15,
      "cases": 600
    },
    "sql": {
      "weight": 0.1,
      "cases": 1000
    }
  },
  "ladder": null
}
```

## Citations / decisions

- Crown: `G_MIN = -ln(0.95)`; ship with margin toward `-ln(0.93)` on both halves.
- Breadth: ≥2 tracks with positive LCB99; per-track regression guard; `TRACK_GAIN_CAP = ln 2`.
- Two serve paths: reads → structured_server `/v1/systemone`; harness → chat + `harness.replay`.
- Init from **champion**, not base. Measure base only as reference.
- Export: merged BF16 safetensors + byte-identical `config.json`. Adapters never ship.
- Champion decay (v2.2.0): prefer fast re-crowns under 36h full / 36h half-life.

## Next actions

1. `opentype-miner data generate --plan --smoke` then full plan
2. `opentype-miner data oracle-chat --in data/public --out data/public/chat`
3. Build local sim headroom vs champion
4. Read-slot SFT until forfeit≈0 + parity pass
5. Harness oracle SFT → on-policy RFT/DPO
6. WiSE-FT α sweep → export → manual submit
