"""Train/serve parity: slot logprobs within BF16 of vLLM + structured_server.

Test #1 before any big run. Compares offline read-head slot distributions to a
live structured_server /v1/systemone for the same request bodies.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import httpx


def max_abs_diff(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


def bf16_tol(ref: list[float]) -> float:
    """Loose BF16 absolute tolerance on probabilities."""
    return max(2e-3, 0.02 * max(ref))


def compare_answers(
    offline: dict[str, Any], served: dict[str, Any]
) -> dict[str, Any]:
    """Compare probability vectors / label_mass between two systemone payloads."""
    failures: list[str] = []
    details: dict[str, Any] = {}
    sa = served.get("answers") or {}
    oa = offline.get("answers") or {}
    sr = (served.get("diagnostics") or {}).get("questions") or served.get("reads") or {}
    or_ = offline.get("reads") or (offline.get("diagnostics") or {}).get("questions") or {}
    for qid in sorted(set(sa) | set(oa)):
        if qid not in sa or qid not in oa:
            failures.append(f"{qid}: missing on one side")
            continue
        s, o = sa[qid], oa[qid]
        if "noul" in s or "noul" in o:
            sp, op = [float(s.get("noul", 0)), 1 - float(s.get("noul", 0))], [
                float(o.get("noul", 0)),
                1 - float(o.get("noul", 0)),
            ]
        else:
            sp = [float(x) for x in (s.get("probabilities") or {}).values()]
            op = [float(x) for x in (o.get("probabilities") or {}).values()]
            if len(sp) != len(op):
                # align by key
                keys = sorted(set(s.get("probabilities", {})) | set(o.get("probabilities", {})))
                sp = [float(s.get("probabilities", {}).get(k, 0)) for k in keys]
                op = [float(o.get("probabilities", {}).get(k, 0)) for k in keys]
        err = max_abs_diff(sp, op)
        tol = bf16_tol(sp)
        lm_s = float((sr.get(qid) or {}).get("label_mass", math.nan))
        lm_o = float((or_.get(qid) or {}).get("label_mass", math.nan))
        details[qid] = {"max_abs": err, "tol": tol, "label_mass_served": lm_s, "label_mass_offline": lm_o}
        if err > tol:
            failures.append(f"{qid}: prob max_abs {err:.4g} > tol {tol:.4g}")
        if math.isfinite(lm_s) and math.isfinite(lm_o) and abs(lm_s - lm_o) > 0.05:
            failures.append(f"{qid}: label_mass drift {lm_s:.4g} vs {lm_o:.4g}")
    return {"ok": not failures, "failures": failures, "details": details}


def call_systemone(url: str, body: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    r = httpx.post(url.rstrip("/") + "/v1/systemone", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def parity_file(
    cases: Path,
    *,
    offline_url: str,
    served_url: str,
    max_cases: int = 32,
) -> dict[str, Any]:
    results = []
    with cases.open() as handle:
        n = 0
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("track") in ("ops", "sql", "paint"):
                continue
            body = row["request"]
            offline = call_systemone(offline_url, body)
            served = call_systemone(served_url, body)
            cmp_ = compare_answers(offline, served)
            results.append({"index": n, "ok": cmp_["ok"], **cmp_})
            n += 1
            if n >= max_cases:
                break
    ok = all(r["ok"] for r in results) if results else False
    return {"ok": ok, "n": len(results), "results": results}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Read-slot train/serve parity check")
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--offline-url", required=True, help="train-time reader base URL")
    p.add_argument("--served-url", required=True, help="vLLM+structured_server base URL")
    p.add_argument("--max-cases", type=int, default=32)
    p.add_argument("--json-out", type=Path)
    args = p.parse_args(argv)
    report = parity_file(
        args.cases,
        offline_url=args.offline_url,
        served_url=args.served_url,
        max_cases=args.max_cases,
    )
    print(json.dumps({"ok": report["ok"], "n": report["n"]}, indent=2))
    if not report["ok"]:
        for r in report["results"]:
            if not r["ok"]:
                print(r["failures"])
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
