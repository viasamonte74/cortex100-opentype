"""Harness SFT: block-diffusion CE on assistant JSON turns only.

Uses DiffusionGemmaForBlockDiffusion: prompt → encoder input_ids, response →
decoder_input_ids canvas (uniform corruption).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from opentype_miner.pins import BASE_REPO, BASE_REVISION, CANVAS
from opentype_miner.train.model_load import (
    as_token_ids,
    attach_lora,
    load_diffusion_gemma,
    load_tokenizer,
    pad_canvas,
)


def load_chat_jsonl(paths: list[Path], max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("messages"):
                    continue
                rows.append(row)
                if max_rows is not None and len(rows) >= max_rows:
                    return rows
    return rows


def split_prompt_canvas(tokenizer: Any, messages: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    """Prompt tokens + clean response canvas tokens (final assistant turn)."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("chat row must end with an assistant turn")
    prompt_msgs = messages[:-1]
    prompt = as_token_ids(
        tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True
        )
    )
    full = as_token_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    )
    resp = full[len(prompt) :]
    return prompt, resp


def train(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset

    from opentype_miner.train.diffusion_loss import corrupt_uniform, diffusion_ce

    class ChatDS(Dataset):
        def __init__(self, rows: list[dict[str, Any]], tokenizer):
            self.rows = rows
            self.tok = tokenizer

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
            prompt, resp = split_prompt_canvas(self.tok, self.rows[i]["messages"])
            canvas = pad_canvas(resp, CANVAS, self.tok.pad_token_id or 0)
            # supervise only real response tokens (not pad)
            mask = [1] * min(len(resp), CANVAS) + [0] * max(0, CANVAS - len(resp))
            return {
                "input_ids": torch.tensor(prompt, dtype=torch.long),
                "decoder_input_ids": torch.tensor(canvas, dtype=torch.long),
                "canvas_mask": torch.tensor(mask, dtype=torch.bool),
            }

    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        # pad prompts to max in batch
        max_p = max(x["input_ids"].numel() for x in batch)
        pad_id = 0
        prompts, attns, decs, masks = [], [], [], []
        for x in batch:
            p = x["input_ids"]
            pad = max_p - p.numel()
            if pad:
                p = torch.cat([p, torch.full((pad,), pad_id, dtype=torch.long)])
                a = torch.cat([torch.ones(x["input_ids"].numel(), dtype=torch.long), torch.zeros(pad, dtype=torch.long)])
            else:
                a = torch.ones(max_p, dtype=torch.long)
            prompts.append(p)
            attns.append(a)
            decs.append(x["decoder_input_ids"])
            masks.append(x["canvas_mask"])
        return {
            "input_ids": torch.stack(prompts),
            "attention_mask": torch.stack(attns),
            "decoder_input_ids": torch.stack(decs),
            "canvas_mask": torch.stack(masks),
        }

    rows = load_chat_jsonl([Path(p) for p in args.data], max_rows=args.max_rows)
    if not rows:
        raise SystemExit("no chat rows; run: opentype-miner data oracle-chat ...")

    model_id = args.model or BASE_REPO
    revision = args.revision or BASE_REVISION
    print(f"loading {model_id}@{revision} via DiffusionGemmaForBlockDiffusion", flush=True)
    # Chat template/tokenizer always from pinned base (same as duel worker).
    print(f"tokenizer {BASE_REPO}@{BASE_REVISION} (serve parity)", flush=True)
    tokenizer = load_tokenizer(BASE_REPO, BASE_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_diffusion_gemma(
        model_id,
        revision,
        device_map=args.device_map,
        freeze_vision=bool(args.freeze_vision),
    )
    model = attach_lora(
        model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    model.train()
    device = next(model.parameters()).device
    ds = ChatDS(rows, tokenizer)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    vocab = int(getattr(getattr(model.config, "text_config", model.config), "vocab_size", 262144))
    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            clean = batch["decoder_input_ids"].to(device)
            canvas_mask = batch["canvas_mask"].to(device)
            noisy, supervised, _t = corrupt_uniform(
                clean, canvas_mask, vocab_size=vocab, eps=args.eps
            )
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                decoder_input_ids=noisy,
            )
            loss = diffusion_ce(out.logits, clean, supervised)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                print(f"epoch={epoch} step={step} loss={float(loss):.4f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "train_meta.json").write_text(
        json.dumps(
            {
                "kind": "harness_block_diffusion_lora",
                "arch": "DiffusionGemmaForBlockDiffusion",
                "base": f"{model_id}@{revision}",
                "steps": step,
                "canvas": CANVAS,
                "eps": args.eps,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved adapter → {out_dir}", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="OpenType harness block-diffusion SFT")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--eps", type=float, default=0.001)
    p.add_argument("--freeze-vision", action="store_true", default=True)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--config", type=Path)
    args = p.parse_args(argv)
    if args.config and args.config.exists():
        cfg = yaml.safe_load(args.config.read_text()) or {}
        for k, v in cfg.items():
            key = k.replace("-", "_")
            if hasattr(args, key):
                setattr(args, key, v)
    train(args)


if __name__ == "__main__":
    main()
