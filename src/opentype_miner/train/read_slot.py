"""Read-slot SFT: soft posterior + label-mass, mirroring structured_server.

Never hard-label AR. Prefers a live reader (vLLM+structured_server) for
distillation / RFT; offline path trains soft CE when the model exposes logits.

Hard priors:
  - import pinned structured_server path
  - 1-step canvas denoise (canvas=256)
  - soft posterior on label tokens; -log(label_mass) penalty
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from opentype_miner.pins import BASE_REPO, BASE_REVISION, CANVAS, assert_structured_server
from opentype_miner.serve.read_protocol import (
    bind_tokenizer,
    jev_to_schema,
    load_structured_server,
    soft_targets_for_gold,
)
from opentype_miner.train.model_load import (
    as_token_ids,
    attach_lora,
    load_diffusion_gemma,
    load_tokenizer,
    pad_canvas,
)


@dataclass
class ReadExample:
    body: dict[str, Any]
    gold: dict[str, list[Any]]
    track: str
    level: int


def load_read_jsonl(paths: list[Path], max_rows: int | None = None) -> list[ReadExample]:
    out: list[ReadExample] = []
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "gold" not in row:
                    continue
                out.append(
                    ReadExample(row["request"], row["gold"], row["track"], int(row["level"]))
                )
                if max_rows is not None and len(out) >= max_rows:
                    return out
    return out


def forfeit_stats(examples: list[ReadExample], predict) -> dict[str, float]:
    """``predict(body) -> (answers, reads)``; measure forfeit rate under scoring rules."""
    from opentype_challenge import scoring
    from opentype_challenge.generator import Gold

    n = forfeits = 0
    loss_sum = 0.0
    for ex in examples:
        answers, reads = predict(ex.body)
        gold = {qid: Gold.from_json(v) for qid, v in ex.gold.items()}
        score = scoring.score_case(gold, answers, reads)
        n += score.decisions
        # forfeit items contribute loss 1.0 each; approximate via loss==decisions when all forfeit
        for qid, g in gold.items():
            item_loss, _ = scoring.decision_loss(
                g, (answers or {}).get(qid), (reads or {}).get(qid)
            )
            if item_loss >= 1.0 - 1e-12:
                forfeits += 1
            loss_sum += item_loss
    return {
        "decisions": n,
        "forfeit_rate": forfeits / max(n, 1),
        "mean_loss": loss_sum / max(n, 1),
    }


def build_slot_targets(ss, tokenizer, ex: ReadExample) -> list[dict[str, Any]]:
    """Per-question soft targets aligned to structured_server label_ids."""
    bind_tokenizer(ss, tokenizer)
    schema = jev_to_schema(ss, ex.body)
    template, slots = ss.template_for(schema, list(ss.SCAFFOLD), "")
    by_id = {q["id"]: q for q in schema["questions"]}
    targets = []
    for q, slot in zip(schema["questions"], slots, strict=True):
        gold = ex.gold[q["id"]]
        probs = soft_targets_for_gold(by_id[q["id"]], gold)
        targets.append(
            {
                "id": q["id"],
                "pos": slot["pos"],
                "label_ids": list(slot["label_ids"]),
                "target": probs,
                "template": template,
            }
        )
    return targets


def train_offline(args: argparse.Namespace) -> None:
    """LoRA fine-tune with soft slot CE on DiffusionGemmaForBlockDiffusion."""
    import torch

    from opentype_miner.train.diffusion_loss import soft_label_ce_from_logits

    assert_structured_server()
    ss = load_structured_server()
    ss.CANVAS_LEN = CANVAS

    paths = [Path(p) for p in args.data]
    examples = load_read_jsonl(paths, max_rows=args.max_rows)
    if not examples:
        raise SystemExit("no read examples with gold")

    model_id = args.model or BASE_REPO
    revision = args.revision or BASE_REVISION
    print(f"loading {model_id}@{revision} via DiffusionGemmaForBlockDiffusion")
    # Duel always uses the pinned base tokenizer/templates (worker assemble). Champion
    # repos often omit a usable tokenizer — never resolve slots with the weight repo's tok.
    print(f"tokenizer {BASE_REPO}@{BASE_REVISION} (serve parity)", flush=True)
    tokenizer = load_tokenizer(BASE_REPO, BASE_REVISION)
    bind_tokenizer(ss, tokenizer)
    model = load_diffusion_gemma(
        model_id,
        revision,
        device_map=args.device_map,
        freeze_vision=bool(args.freeze_vision),
        quantize=args.quantize,
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    model = attach_lora(
        model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    model.train()
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    rng = random.Random(args.seed)
    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for idx in order:
            ex = examples[idx]
            try:
                slots = build_slot_targets(ss, tokenizer, ex)
            except Exception as err:  # noqa: BLE001 — skip schema that does not fit canvas
                print(f"skip schema: {err}", flush=True)
                continue
            try:
                sys_text = ss.system_text(jev_to_schema(ss, ex.body))
                messages = [
                    {"role": "system", "content": sys_text},
                    {"role": "user", "content": ex.body["state"]},
                ]
                prompt = as_token_ids(
                    tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    )
                )
                if args.max_prompt_len and len(prompt) > args.max_prompt_len:
                    # keep the tail (criteria/questions live near the end of the prompt)
                    prompt = prompt[-args.max_prompt_len :]

                # 1-step seeded canvas (serve path): template + noise at label slots, pad to 256
                template = slots[0]["template"]
                canvas = list(template) + [ss.TURN_CLOSE]
                canvas = pad_canvas(canvas, CANVAS, ss.PAD)
                for s in slots:
                    if s["pos"] < CANVAS:
                        canvas[s["pos"]] = rng.randrange(ss.VOCAB)

                input_ids = torch.tensor([prompt], device=device)
                decoder_input_ids = torch.tensor([canvas], device=device)
                attn = torch.ones_like(input_ids)
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    decoder_input_ids=decoder_input_ids,
                )
                logits = out.logits[0]  # [canvas_len, V]
                loss = torch.zeros((), device=device)
                mass_pen = torch.zeros((), device=device)
                for s in slots:
                    pos = s["pos"]
                    if pos >= logits.shape[0]:
                        continue
                    label_ids = torch.tensor(s["label_ids"], device=device)
                    target = torch.tensor(s["target"], device=device, dtype=logits.dtype)
                    ce, mass = soft_label_ce_from_logits(logits[pos], label_ids, target)
                    loss = loss + ce
                    mass_pen = mass_pen + torch.relu(
                        torch.tensor(0.55, device=device) - mass
                    )
                loss = (
                    loss / max(len(slots), 1)
                    + args.mass_weight * mass_pen / max(len(slots), 1)
                )
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                step += 1
                if step % args.log_every == 0:
                    print(f"epoch={epoch} step={step} loss={float(loss):.4f}", flush=True)
            except Exception as err:  # noqa: BLE001
                print(f"skip step: {err}", flush=True)
                continue
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    if step == 0:
        raise SystemExit("no training steps ran (all examples skipped); check tokenizer/slots")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta = {
        "kind": "read_slot_lora",
        "arch": "DiffusionGemmaForBlockDiffusion",
        "base": f"{model_id}@{revision}",
        "steps": step,
        "canvas": CANVAS,
        "structured_server": str(assert_structured_server()),
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved adapter → {out_dir}", flush=True)


def distill_from_reader(args: argparse.Namespace) -> None:
    """Collect soft targets from a live structured_server (champion or student)."""
    import httpx

    paths = [Path(p) for p in args.data]
    examples = load_read_jsonl(paths, max_rows=args.max_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as handle, httpx.Client(timeout=300.0) as client:
        for ex in examples:
            r = client.post(args.reader.rstrip("/") + "/v1/systemone", json=ex.body)
            if r.status_code >= 400:
                continue
            payload = r.json()
            handle.write(
                json.dumps(
                    {
                        "track": ex.track,
                        "level": ex.level,
                        "request": ex.body,
                        "gold": ex.gold,
                        "teacher_answers": payload.get("answers"),
                        "teacher_reads": (payload.get("diagnostics") or {}).get("questions"),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            n += 1
    print(f"wrote {n} teacher rows → {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="OpenType read-slot training")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="offline LoRA soft-slot training")
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model", default=None, help="HF id; default base (prefer champion)")
    t.add_argument("--revision", default=None)
    t.add_argument("--device-map", default="auto")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--max-steps", type=int)
    t.add_argument("--max-rows", type=int)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--lora-r", type=int, default=64)
    t.add_argument("--lora-alpha", type=int, default=128)
    t.add_argument("--lora-dropout", type=float, default=0.05)
    t.add_argument("--mass-weight", type=float, default=0.2)
    t.add_argument("--freeze-vision", action="store_true", default=True)
    t.add_argument("--log-every", type=int, default=20)
    t.add_argument("--quantize", choices=["4bit", "8bit", "bf16"], default="4bit",
                   help="base-weight precision (8bit ~91GiB weights alone on this MoE; prefer 4bit)")
    t.add_argument("--load-in-4bit", action="store_true",
                   help="alias for --quantize 4bit")
    t.add_argument("--no-load-in-4bit", action="store_true",
                   help="alias for --quantize bf16 (ignored if --quantize set explicitly)")
    t.add_argument("--gradient-checkpointing", action="store_true", default=True)
    t.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    t.add_argument("--max-prompt-len", type=int, default=2048,
                   help="truncate prompt tokens (longctx needs a bigger GPU/EP setup)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--config", type=Path, help="optional YAML overrides")

    d = sub.add_parser("distill", help="dump teacher soft answers from a live reader")
    d.add_argument("--data", nargs="+", required=True)
    d.add_argument("--reader", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--max-rows", type=int)

    f = sub.add_parser("forfeit-check", help="measure forfeit rate via a reader URL")
    f.add_argument("--data", nargs="+", required=True)
    f.add_argument("--reader", required=True)
    f.add_argument("--max-rows", type=int, default=200)

    args = p.parse_args(argv)
    # CLI aliases for quantize
    if getattr(args, "load_in_4bit", False):
        args.quantize = "4bit"
    elif getattr(args, "no_load_in_4bit", False) and getattr(args, "quantize", None) == "8bit":
        # only treat as bf16 if user didn't pass an explicit non-default via env; keep simple
        pass
    if args.cmd == "train":
        if args.config and args.config.exists():
            cfg = yaml.safe_load(args.config.read_text()) or {}
            for k, v in cfg.items():
                key = k.replace("-", "_")
                if hasattr(args, key) and getattr(args, key) is None:
                    setattr(args, key, v)
        train_offline(args)
    elif args.cmd == "distill":
        distill_from_reader(args)
    else:
        import httpx

        examples = load_read_jsonl([Path(p) for p in args.data], max_rows=args.max_rows)

        def predict(body: dict[str, Any]):
            r = httpx.post(args.reader.rstrip("/") + "/v1/systemone", json=body, timeout=300)
            r.raise_for_status()
            payload = r.json()
            return payload.get("answers"), (payload.get("diagnostics") or {}).get("questions")

        print(json.dumps(forfeit_stats(examples, predict), indent=2))


if __name__ == "__main__":
    main()
