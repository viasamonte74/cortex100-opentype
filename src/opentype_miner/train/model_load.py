"""Load DiffusionGemma the way the duel serves it (not AutoModelForCausalLM)."""

from __future__ import annotations

import os
from typing import Any, Literal

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, BitsAndBytesConfig, DiffusionGemmaForBlockDiffusion

from opentype_miner.pins import LORA_TARGET_MODULES

QuantMode = Literal["4bit", "8bit", "bf16"]


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def load_tokenizer(model_id: str, revision: str | None) -> Any:
    return AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        token=hf_token(),
        trust_remote_code=True,
    )


def load_diffusion_gemma(
    model_id: str,
    revision: str | None,
    *,
    device_map: str = "auto",
    freeze_vision: bool = True,
    freeze_router: bool = True,
    quantize: QuantMode = "4bit",
    load_in_4bit: bool | None = None,  # backward compat
    gradient_checkpointing: bool = True,
) -> DiffusionGemmaForBlockDiffusion:
    if load_in_4bit is True:
        quantize = "4bit"
    elif load_in_4bit is False and quantize == "4bit":
        quantize = "bf16"

    kwargs: dict[str, Any] = {
        "revision": revision,
        "device_map": device_map,
        "token": hf_token(),
        "trust_remote_code": True,
    }
    if quantize == "4bit":
        # Day-1 path on 1×96GB: ~65GiB with LoRA+acts; ~30GiB headroom for rank/context.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantize == "8bit":
        # Weights alone ~91GiB on this MoE — usually no room for forward/backward.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    else:
        kwargs["dtype"] = torch.bfloat16

    print(f"quantize={quantize}", flush=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_id, **kwargs)

    for name, param in model.named_parameters():
        lname = name.lower()
        if freeze_router and "router" in lname:
            param.requires_grad = False
        if freeze_vision and (
            "vision" in lname or "multi_modal" in lname or "mm_encoder" in lname
        ):
            param.requires_grad = False

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print("gradient checkpointing: on", flush=True)
    return model


# Vision q_proj/etc. are Gemma4ClippableLinear (unsupported by PEFT).
# Decoder-only keeps optimizer/activation pressure lower on 1×GPU; encoder stays frozen base.
_TEXT_LORA_TARGET_REGEX = (
    r".*\.decoder\.layers\.\d+\."
    r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$"
)


def attach_lora(
    model: DiffusionGemmaForBlockDiffusion,
    *,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> Any:
    """LoRA on diffusion-decoder attn + dense MLP only (never vision ClippableLinear)."""
    _ = LORA_TARGET_MODULES
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=_TEXT_LORA_TARGET_REGEX,
        bias="none",
    )
    peft_model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"LoRA attached: trainable={trainable:,} / total={total:,}", flush=True)
    if trainable == 0:
        raise RuntimeError("LoRA matched no modules; check _TEXT_LORA_TARGET_REGEX")
    return peft_model


def pad_canvas(ids: list[int], length: int, pad_id: int = 0) -> list[int]:
    ids = ids[:length]
    if len(ids) < length:
        ids = ids + [pad_id] * (length - len(ids))
    return ids


def as_token_ids(value: Any) -> list[int]:
    """Normalize tokenizer / chat-template outputs to a flat list of ints.

    Newer transformers may return a BatchEncoding (Mapping) whose iteration yields
    keys like ``input_ids``, not the token ids — that caused:
    ``invalid literal for int() with base 10: 'input_ids'``.
    """
    if value is None:
        return []
    if isinstance(value, dict) or hasattr(value, "keys"):
        try:
            value = value["input_ids"]
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return [int(t) for t in value]
