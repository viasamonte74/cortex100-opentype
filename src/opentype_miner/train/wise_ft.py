"""WiSE-FT: interpolate merged student weights toward the champion.

α-sweep after each training round. Prefer sim LCB over train loss when picking α.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def interpolate(
    champion_dir: Path,
    student_dir: Path,
    out_dir: Path,
    alpha: float,
) -> None:
    """out = (1-α)·champion + α·student for every overlapping floating tensor.

    Expects sharded or single safetensors + config.json already present in champion_dir.
    Copies non-tensor files from champion (including byte-identical config.json).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Discover weight files
    champ_files = sorted(champion_dir.glob("*.safetensors"))
    stud_files = {p.name: p for p in student_dir.glob("*.safetensors")}
    if not champ_files:
        raise SystemExit(f"no safetensors in {champion_dir}")
    for cpath in champ_files:
        spath = stud_files.get(cpath.name)
        if spath is None:
            # copy champion shard untouched
            data = load_file(str(cpath))
            save_file(data, str(out_dir / cpath.name))
            continue
        c = load_file(str(cpath))
        s = load_file(str(spath))
        mixed = {}
        for k, cv in c.items():
            if k in s and cv.dtype.is_floating_point and s[k].shape == cv.shape:
                mixed[k] = ((1.0 - alpha) * cv.float() + alpha * s[k].float()).to(cv.dtype)
            else:
                mixed[k] = cv
        save_file(mixed, str(out_dir / cpath.name))
    # Copy sidecar files from champion (config must stay byte-identical to base)
    for name in ("config.json", "model.safetensors.index.json"):
        src = champion_dir / name
        if src.exists():
            (out_dir / name).write_bytes(src.read_bytes())
    (out_dir / "wise_ft.json").write_text(
        json.dumps(
            {
                "alpha": alpha,
                "champion": str(champion_dir),
                "student": str(student_dir),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"WiSE-FT α={alpha} → {out_dir}")


def sweep(args: argparse.Namespace) -> None:
    for alpha in args.alphas:
        out = Path(args.out) / f"alpha_{alpha:.3f}".replace(".", "p")
        interpolate(Path(args.champion), Path(args.student), out, alpha)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="WiSE-FT α interpolate toward champion")
    p.add_argument("--champion", type=Path, required=True)
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7, 0.85, 1.0],
    )
    args = p.parse_args(argv)
    sweep(args)


if __name__ == "__main__":
    main()
