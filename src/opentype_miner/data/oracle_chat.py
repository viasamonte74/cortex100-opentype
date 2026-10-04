"""Turn harness oracle rows into chat training examples via harness.chat_messages."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from opentype_challenge import harness, tracks


def conversations(row: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """One training example per turn: messages including the assistant output."""
    if not row.get("oracle"):
        return []
    env, outputs, seen = tracks.ENVS[row["track"]], iter(row["oracle"]), []

    async def generate(messages: list[dict[str, Any]], _seed: int) -> str:
        output = next(outputs)
        seen.append(messages + [{"role": "assistant", "content": output}])
        return output

    asyncio.run(harness.run_episode(env, row["request"], generate))
    return seen


def iter_chat_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("track") not in tracks.ENVS:
                continue
            for example in conversations(row):
                yield {
                    "track": row["track"],
                    "family": row.get("family"),
                    "level": row.get("level"),
                    "messages": example,
                }


def convert(in_path: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as out:
        for row in iter_chat_rows(in_path):
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
            n += 1
    return n


def convert_dir(in_dir: Path, out_dir: Path, pattern: str = "*.jsonl") -> dict[str, int]:
    counts: dict[str, int] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob(pattern)):
        if path.name.startswith("chat_"):
            continue
        # Only harness files carry oracles.
        peek = path.open().readline()
        if not peek:
            continue
        if "oracle" not in peek and path.stem.split("_")[0] not in tracks.ENVS:
            # cheap filter; still try convert if filename matches
            if not any(path.name.startswith(t) for t in tracks.ENVS):
                continue
        out = out_dir / f"chat_{path.name}"
        counts[out.name] = convert(path, out)
        print(f"wrote {counts[out.name]} → {out}")
    return counts


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Oracle JSONL → chat SFT JSONL")
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.inp.is_dir():
        convert_dir(args.inp, args.out)
    else:
        n = convert(args.inp, args.out)
        print(f"wrote {n} → {args.out}")


if __name__ == "__main__":
    main()
