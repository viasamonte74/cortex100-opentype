"""Write / refresh RECON.md with pins, citations, and live status."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
from opentype_challenge import __version__ as challenge_version
from opentype_challenge import pins as challenge_pins

from opentype_miner import __version__ as miner_version
from opentype_miner.paths import RECON, STRUCTURED_SERVER, VENDOR
from opentype_miner.pins import (
    BASE_CONFIG_SHA256,
    BASE_REPO,
    BASE_REVISION,
    CANVAS,
    STRUCTURED_SERVER_SHA256,
    STRUCTURED_SERVER_URL,
    VLLM_IMAGE,
    assert_structured_server,
    sha256_file,
)


def fetch_structured_server() -> Path:
    VENDOR.mkdir(parents=True, exist_ok=True)
    r = httpx.get(STRUCTURED_SERVER_URL, timeout=60, follow_redirects=True)
    r.raise_for_status()
    STRUCTURED_SERVER.write_bytes(r.content)
    assert_structured_server()
    return STRUCTURED_SERVER


def live_status(api: str | None) -> dict:
    if not api:
        return {}
    r = httpx.get(api.rstrip("/") + "/v1/status", timeout=60)
    r.raise_for_status()
    return r.json()


def write_recon(api: str | None = None, champion_override: str | None = None) -> Path:
    try:
        ss_path = assert_structured_server()
        ss_ok = True
        ss_sha = sha256_file(ss_path)
    except Exception as err:  # noqa: BLE001
        ss_ok = False
        ss_sha = str(err)
    status = live_status(api) if api else {}
    champion = status.get("champion") or {}
    if champion_override:
        champion = {"repo": champion_override, **champion}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = f"""# RECON — OpenType miner

Updated: {now}

## Pins (code is source of truth)

| Item | Value |
| --- | --- |
| challenge package | `{challenge_version}` (target **2.2.0**) |
| miner package | `{miner_version}` |
| base | `{BASE_REPO}@{BASE_REVISION}` |
| config.json sha256 | `{BASE_CONFIG_SHA256}` |
| structured_server URL | `{STRUCTURED_SERVER_URL}` |
| structured_server sha256 | `{STRUCTURED_SERVER_SHA256}` |
| local structured_server | `{STRUCTURED_SERVER}` match={ss_ok} actual=`{ss_sha}` |
| vLLM image | `{VLLM_IMAGE}` |
| canvas | `{CANVAS}` |
| max_model_len | `131072` |

## Live status

```json
{json.dumps({"champion": champion, "plan": status.get("plan"), "ladder": status.get("ladder")}, indent=2, default=str)}
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
"""
    RECON.write_text(text)
    print(f"wrote {RECON}")
    return RECON


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Refresh RECON.md")
    p.add_argument("--api", help="challenge API base, e.g. https://host/challenge/opentype")
    p.add_argument("--fetch-structured-server", action="store_true")
    p.add_argument("--champion", help="override champion repo note")
    args = p.parse_args(argv)
    if args.fetch_structured_server:
        fetch_structured_server()
        print(f"pinned {STRUCTURED_SERVER}")
    write_recon(args.api, args.champion)


if __name__ == "__main__":
    main()
