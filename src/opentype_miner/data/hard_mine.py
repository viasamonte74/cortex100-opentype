"""Active learning: upsample high-loss / high-gap cases into on_policy mix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from opentype_miner.paths import MIX_YAML, ON_POLICY


def load_mix(path: Path = MIX_YAML) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def mine(
    case_scores: Path,
    cases: Path,
    out: Path,
    *,
    mix_path: Path = MIX_YAML,
) -> dict[str, int]:
    """``case_scores`` JSONL: {index, track, challenger_loss, champion_loss, ...}.

    Writes upsampled copies of the matching ``cases`` JSONL rows into ``out``.
    """
    cfg = load_mix(mix_path)["active_learning"]
    scores = [json.loads(line) for line in case_scores.read_text().splitlines() if line.strip()]
    rows = [json.loads(line) for line in cases.read_text().splitlines() if line.strip()]
    if len(scores) != len(rows):
        # allow scores keyed by index
        by_idx = {int(s["index"]): s for s in scores}
        paired = [(rows[i], by_idx[i]) for i in range(len(rows)) if i in by_idx]
    else:
        paired = list(zip(rows, scores, strict=True))

    by_loss = sorted(paired, key=lambda x: float(x[1]["challenger_loss"]), reverse=True)
    by_gap = sorted(
        paired,
        key=lambda x: float(x[1]["challenger_loss"]) - float(x[1]["champion_loss"]),
        reverse=True,
    )
    n = len(paired)
    top_n = max(1, int(n * float(cfg["top_loss_frac"])))
    gap_n = max(1, int(n * float(cfg["gap_frac"])))
    chosen: list[tuple[dict, float]] = []
    seen: set[int] = set()
    for i, (row, sc) in enumerate(by_loss[:top_n]):
        key = id(row)
        if key in seen:
            continue
        seen.add(key)
        chosen.append((row, float(cfg["min_upsample"])))
    for row, sc in by_gap[:gap_n]:
        key = id(row)
        gap = float(sc["challenger_loss"]) - float(sc["champion_loss"])
        weight = min(float(cfg["max_upsample"]), max(float(cfg["min_upsample"]), 2.0 + gap))
        if key in seen:
            # bump existing
            for j, (r, w) in enumerate(chosen):
                if id(r) == key:
                    chosen[j] = (r, max(w, weight))
                    break
        else:
            seen.add(key)
            chosen.append((row, weight))

    out.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with out.open("w") as handle:
        for row, weight in chosen:
            copies = max(1, int(round(weight)))
            for _ in range(copies):
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                counts[row.get("track", "?")] += 1
    summary = dict(counts)
    print(json.dumps({"out": str(out), "rows": sum(counts.values()), "by_track": summary}))
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Hard-example mine from sim scores")
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ON_POLICY / "hard_mine.jsonl")
    args = p.parse_args(argv)
    mine(args.scores, args.cases, args.out)


if __name__ == "__main__":
    main()
