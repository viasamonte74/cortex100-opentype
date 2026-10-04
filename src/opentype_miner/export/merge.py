"""Export = merged BF16 + byte-identical config.json; adapters never ship."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from opentype_miner.pins import BASE_CONFIG_SHA256, BASE_REPO, BASE_REVISION


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fetch_base_config(out: Path) -> Path:
    path = Path(
        hf_hub_download(
            BASE_REPO,
            "config.json",
            revision=BASE_REVISION,
            local_dir=str(out / "_base_cfg"),
        )
    )
    digest = sha256_file(path)
    if digest != BASE_CONFIG_SHA256:
        raise RuntimeError(f"base config.json sha {digest} != {BASE_CONFIG_SHA256}")
    return path


def merge_and_export(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import DiffusionGemmaForBlockDiffusion

    from opentype_miner.train.model_load import hf_token

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base_id = args.base or BASE_REPO
    revision = args.revision or BASE_REVISION
    print(f"loading base {base_id}@{revision}")
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        base_id,
        revision=revision,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
        token=hf_token(),
    )
    print(f"merging adapter {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    # Save weights only (BF16)
    model.save_pretrained(out, safe_serialization=True)
    # Force byte-identical base config.json
    cfg = fetch_base_config(out)
    target_cfg = out / "config.json"
    shutil.copyfile(cfg, target_cfg)
    digest = sha256_file(target_cfg)
    if digest != BASE_CONFIG_SHA256:
        raise RuntimeError("exported config.json is not byte-identical to base")
    # Tokenizer files are ignored by the duel (taken from base) — do not rely on shipping them.
    # Still write a small export manifest for the miner.
    files = {
        p.name: sha256_file(p)
        for p in sorted(out.iterdir())
        if p.suffix == ".safetensors" or p.name.endswith(".safetensors") or p.name in {
            "config.json",
            "model.safetensors.index.json",
        }
    }
    # Also hash sharded
    for p in sorted(out.glob("*.safetensors")):
        files[p.name] = sha256_file(p)
    if (out / "model.safetensors.index.json").exists():
        files["model.safetensors.index.json"] = sha256_file(out / "model.safetensors.index.json")
    files["config.json"] = digest
    manifest = {
        "repo_hint": args.hf_repo,
        "base": f"{BASE_REPO}@{BASE_REVISION}",
        "config_sha256": digest,
        "files": files,
        "note": "Push only safetensors (+ index) and config.json. Adapters must not ship.",
    }
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"export ready → {out}")
    print("Submit with: opentype-challenge miner submit --repo <you/model> --revision <40hex> ...")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Merge LoRA and export duel-legal weights")
    p.add_argument("--adapter", required=True, help="PEFT adapter directory")
    p.add_argument("--out", required=True)
    p.add_argument("--base", default=None, help="champion or base HF id used at train init")
    p.add_argument("--revision", default=None)
    p.add_argument("--hf-repo", default=None, help="optional name for the public push target")
    args = p.parse_args(argv)
    merge_and_export(args)


if __name__ == "__main__":
    main()
