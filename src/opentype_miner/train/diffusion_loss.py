"""Block-diffusion training physics (Transformers day-1 path).

Mirrors NeMo DiffusionGemmaSFTRecipe at a high level:
  - uniform-random corruption of supervised canvas tokens (no [MASK])
  - flat CE on corrupted positions
  - optional self-conditioning placeholder (two-pass Analog-Bits)

When the HF DiffusionGemma forward API exposes native diffusion losses, prefer those
and keep this module as the fallback / unit-testable reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class DiffusionBatch:
    input_ids: torch.Tensor  # [B, T] clean tokens (prompt + response)
    attention_mask: torch.Tensor
    # True on supervised canvas positions (assistant response region)
    canvas_mask: torch.Tensor
    # Optional: prompt length per row for encoder AR aux loss
    prompt_lens: torch.Tensor | None = None


def corrupt_uniform(
    input_ids: torch.Tensor,
    canvas_mask: torch.Tensor,
    *,
    vocab_size: int,
    eps: float = 0.001,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (noisy_ids, supervised_mask, t).

    t ~ U(eps, 1); each canvas token is replaced with U{0..V-1} with probability t.
    """
    device = input_ids.device
    bsz = input_ids.shape[0]
    t = torch.empty(bsz, device=device).uniform_(eps, 1.0, generator=generator)
    # Broadcast t to token positions
    t_tok = t[:, None].expand_as(input_ids)
    rand = torch.rand(input_ids.shape, device=device, generator=generator)
    replace = canvas_mask & (rand < t_tok)
    noise = torch.randint(
        0, vocab_size, input_ids.shape, device=device, generator=generator, dtype=input_ids.dtype
    )
    noisy = torch.where(replace, noise, input_ids)
    return noisy, replace, t


def diffusion_ce(
    logits: torch.Tensor,
    clean_ids: torch.Tensor,
    supervised_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean CE over supervised (corrupted) positions."""
    if supervised_mask.dtype != torch.bool:
        supervised_mask = supervised_mask.bool()
    if not supervised_mask.any():
        return logits.sum() * 0.0
    flat_logits = logits[supervised_mask]
    flat_targets = clean_ids[supervised_mask]
    return F.cross_entropy(flat_logits, flat_targets)


def encoder_ar_ce(
    logits: torch.Tensor,
    clean_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
) -> torch.Tensor:
    """Optional causal CE on prompt/encoder tokens (shift-by-1)."""
    if prompt_mask[..., 1:].sum() == 0:
        return logits.sum() * 0.0
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = clean_ids[..., 1:].contiguous()
    shift_mask = prompt_mask[..., 1:].contiguous()
    return F.cross_entropy(shift_logits[shift_mask], shift_labels[shift_mask])


def soft_label_ce_from_logits(
    slot_logits: torch.Tensor,
    label_ids: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft CE over a small set of label token ids at one slot.

    slot_logits: [V] or [B, V]
    label_ids: [L] or [B, L]
    target: [L] or [B, L] probabilities
    Returns (ce, label_mass).
    """
    if slot_logits.dim() == 1:
        slot_logits = slot_logits.unsqueeze(0)
        label_ids = label_ids.unsqueeze(0)
        target = target.unsqueeze(0)
    gathered = torch.gather(slot_logits, -1, label_ids.long())
    log_z = torch.logsumexp(gathered, dim=-1, keepdim=True)
    log_p = gathered - log_z
    ce = -(target * log_p).sum(dim=-1).mean()
    # label_mass ≈ sum softmax mass on labels if logits are full-vocab unnormalized;
    # with only label logits this is 1. Prefer full-vocab when available.
    full_log_z = torch.logsumexp(slot_logits, dim=-1)
    label_mass = torch.exp(log_z.squeeze(-1) - full_log_z).mean()
    return ce, label_mass


def build_canvas_mask_from_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    max_length: int,
    canvas_length: int = 256,
) -> DiffusionBatch:
    """Tokenize a chat example; supervise only the final assistant turn (truncated/padded)."""
    prompt_msgs = messages[:-1]
    assert messages[-1]["role"] == "assistant"
    prompt_ids = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True
    )
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids["input_ids"]
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if isinstance(full_ids, dict):
        full_ids = full_ids["input_ids"]
    prompt_ids = list(map(int, prompt_ids))
    full_ids = list(map(int, full_ids))
    # Cap response canvas
    resp = full_ids[len(prompt_ids) :]
    resp = resp[:canvas_length]
    ids = (prompt_ids + resp)[:max_length]
    mask = [0] * len(prompt_ids) + [1] * len(resp)
    mask = mask[: len(ids)]
    pad = max_length - len(ids)
    if pad > 0:
        pad_id = tokenizer.pad_token_id or 0
        ids = ids + [pad_id] * pad
        mask = mask + [0] * pad
        attn = [1] * (max_length - pad) + [0] * pad
    else:
        attn = [1] * max_length
    t_ids = torch.tensor([ids], dtype=torch.long)
    t_attn = torch.tensor([attn], dtype=torch.long)
    t_canvas = torch.tensor([mask], dtype=torch.bool)
    t_plen = torch.tensor([min(len(prompt_ids), max_length)], dtype=torch.long)
    return DiffusionBatch(t_ids, t_attn, t_canvas, t_plen)
