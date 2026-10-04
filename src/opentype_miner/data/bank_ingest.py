"""Ingest revealed window banks into closed/ and carve a private_proxy holdout."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import httpx
from opentype_challenge import bank as bank_mod

from opentype_miner.paths import CLOSED, PRIVATE_PROXY


def fetch_bank(api: str, window: int) -> bank_mod.Bank:
    rows: list[Any] = []
    api = api.rstrip("/")
    while True:
        page = httpx.get(
            f"{api}/v1/windows/{window}/bank", params={"offset": len(rows)}, timeout=60
        ).json()
        rows += page["items"]
        if not page["items"] or len(rows) >= page["total"]:
            return bank_mod.Bank.from_json(rows)


def split_proxy(
    items: list[dict[str, Any]],
    *,
    proxy_frac: float = 0.15,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold out a private-proxy split (first-class eval, never train)."""
    rng = random.Random(f"proxy|{seed}")
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_kind.setdefault(item["kind"], []).append(item)
    train: list[dict[str, Any]] = []
    proxy: list[dict[str, Any]] = []
    for kind, rows in sorted(by_kind.items()):
        rng.shuffle(rows)
        n_proxy = max(1, int(round(len(rows) * proxy_frac))) if rows else 0
        if len(rows) <= 2:
            n_proxy = 0  # keep tiny kinds for training
        proxy.extend(rows[:n_proxy])
        train.extend(rows[n_proxy:])
    return train, proxy


def save_bank(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2, sort_keys=True) + "\n")


def ingest(
    *,
    api: str | None,
    window: int | None,
    bank_file: Path | None,
    out_dir: Path = CLOSED,
    proxy_dir: Path = PRIVATE_PROXY,
    proxy_frac: float = 0.15,
    seed: int = 0,
) -> dict[str, Any]:
    if bank_file:
        items = json.loads(bank_file.read_text())
        digest = bank_mod.bank_digest(
            [bank_mod.BankItem(i["kind"], i["key"], i["payload"]) for i in items]
        )
    else:
        if not api or window is None:
            raise SystemExit("pass --bank-file or --api and --window")
        b = fetch_bank(api, window)
        items = [
            {"kind": i.kind, "key": i.key, "payload": i.payload} for i in b.items
        ]
        digest = b.digest
    train, proxy = split_proxy(items, proxy_frac=proxy_frac, seed=seed)
    tag = f"window_{window}" if window is not None else bank_file.stem  # type: ignore[union-attr]
    train_path = out_dir / f"{tag}.json"
    proxy_path = proxy_dir / f"{tag}_proxy.json"
    save_bank(train, train_path)
    save_bank(proxy, proxy_path)
    meta = {
        "digest": digest,
        "n_train": len(train),
        "n_proxy": len(proxy),
        "train": str(train_path),
        "proxy": str(proxy_path),
    }
    print(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Ingest revealed OpenType banks")
    p.add_argument("--api")
    p.add_argument("--window", type=int)
    p.add_argument("--bank-file", type=Path)
    p.add_argument("--proxy-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    ingest(
        api=args.api,
        window=args.window,
        bank_file=args.bank_file,
        proxy_frac=args.proxy_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
