# Ablations ritual

Optimize **simulator composite LCB**, not train loss. Kill steps that do not move sim LCB.

| Date | Change | Sim g / g_LCB | Halves | Track gains | Private-proxy gap | Keep? | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | baseline champion | | | | | — | |
| | | | | | | | |

## Kill criteria

- No movement in composite `g_LCB` after a full round → revert.
- Any track with ≥30 pairs regressing on sim → do not ship.
- Private-proxy gap > `mix.yaml ship_gate.private_proxy_max_gap` → more sealed/prose/story aug, less public template.
- Forfeit rate on reads not ≈ 0 → stop harness scaling; fix read-slot + parity first.
