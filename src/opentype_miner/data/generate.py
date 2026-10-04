"""Generate public training JSONL via the challenge generators (disjoint seeds)."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from opentype_challenge import bank, generator, harness, tracks
from opentype_challenge.generator import Case

from opentype_miner.paths import PUBLIC


def _oracle(case: Case) -> list[str] | None:
    policy = getattr(tracks.BUILDERS[case.track], "reference_policy", None)
    if policy is None:
        return None

    async def generate(messages: list[dict[str, Any]], _seed: int) -> str:
        return str(policy(messages))

    import asyncio

    try:
        return asyncio.run(harness.run_episode(tracks.ENVS[case.track], case.body, generate))
    except ValueError:
        return None


def _public_cases(track: str, level: int | None, n: int, seed: int) -> Iterator[Case]:
    builder = tracks.BUILDERS[track]
    levels = builder.buildable(bank.EMPTY_BANK)
    if level is not None and level not in levels:
        raise SystemExit(f"{track} builds levels {list(levels)} without a teacher bank")
    for index in range(n):
        rng = random.Random(f"opentype-miner|{seed}|{track}|{level or '*'}|{index}")
        yield builder.make_case(rng, level or rng.choice(levels), bank.EMPTY_BANK)


def iter_rows(track: str, level: int | None, n: int, seed: int, family: str | None) -> Iterator[dict]:
    if track == "decisions":
        if level is None:
            raise SystemExit("decisions needs --level")
        cases: Iterator[Case] = iter(generator.generate(family, level, n, seed))
    else:
        cases = _public_cases(track, level, n, seed)
    for case in cases:
        row: dict[str, Any] = {
            "track": case.track,
            "family": case.family,
            "level": case.level,
            "request": case.body,
        }
        if case.track in tracks.ENVS:
            row["oracle"] = _oracle(case)
        else:
            row["gold"] = {qid: g.to_json() for qid, g in case.gold.items()}
        yield row


def write_jsonl(path: Path, rows: Iterator[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
            n += 1
    return n


DEFAULT_PLAN = [
    # (track, level, n, seed_offset, filename)
    ("decisions", 1, 20_000, 1, "decisions_l1.jsonl"),
    ("decisions", 2, 20_000, 2, "decisions_l2.jsonl"),
    ("decisions", 3, 40_000, 3, "decisions_l3.jsonl"),
    ("decisions", 4, 40_000, 4, "decisions_l4.jsonl"),
    ("decisions", 5, 30_000, 5, "decisions_l5.jsonl"),
    ("decisions", 6, 20_000, 6, "decisions_l6.jsonl"),
    ("decisions", 7, 15_000, 7, "decisions_l7.jsonl"),
    ("decisions", 8, 15_000, 8, "decisions_l8.jsonl"),
    ("longctx", 1, 2_000, 11, "longctx_l1.jsonl"),
    ("longctx", 2, 2_000, 12, "longctx_l2.jsonl"),
    ("longctx", 3, 1_500, 13, "longctx_l3.jsonl"),
    ("ops", None, 20_000, 21, "ops.jsonl"),
    ("sql", None, 20_000, 31, "sql.jsonl"),
    ("paint", None, 15_000, 41, "paint_spec.jsonl"),
]


def generate_bucket(
    out_dir: Path,
    *,
    seed: int = 1,
    smoke: bool = False,
    tracks_filter: set[str] | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for track, level, n, seed_off, name in DEFAULT_PLAN:
        if tracks_filter and track not in tracks_filter:
            continue
        n_use = min(n, 32) if smoke else n
        path = out_dir / name
        count = write_jsonl(path, iter_rows(track, level, n_use, seed + seed_off, None))
        counts[name] = count
        print(f"wrote {count} → {path}")
    return counts


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate public OpenType training JSONL")
    p.add_argument("--out", type=Path, default=PUBLIC)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--track", choices=["decisions", "longctx", "ops", "sql", "paint"])
    p.add_argument("--level", type=int)
    p.add_argument("--n", type=int)
    p.add_argument("--family", default=None)
    p.add_argument("--smoke", action="store_true", help="tiny counts for pipeline tests")
    p.add_argument("--plan", action="store_true", help="write the default multi-file plan")
    args = p.parse_args(argv)
    if args.plan or (args.track is None and args.n is None):
        filt = {args.track} if args.track else None
        generate_bucket(args.out, seed=args.seed, smoke=args.smoke, tracks_filter=filt)
        return
    if args.track is None or args.n is None:
        raise SystemExit("pass --plan, or --track and --n")
    name = f"{args.track}" + (f"_l{args.level}" if args.level else "") + ".jsonl"
    path = args.out / name
    n = write_jsonl(path, iter_rows(args.track, args.level, args.n, args.seed, args.family))
    print(f"wrote {n} → {path}")


if __name__ == "__main__":
    main()
