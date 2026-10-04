"""Import the pinned structured_server and expose slot / soft-posterior helpers.

Train/serve parity rule: every read training example must go through the same
Jev→schema→template→slot resolution the duel worker uses.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

from opentype_miner.pins import CANVAS, assert_structured_server
from opentype_miner.paths import STRUCTURED_SERVER


def load_structured_server(path: Path | None = None) -> types.ModuleType:
    path = assert_structured_server(path or STRUCTURED_SERVER)
    name = "_opentype_structured_server"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    # Match worker canvas before module-level defaults are used by callers.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.CANVAS_LEN = CANVAS
    return module


def bind_tokenizer(ss: types.ModuleType, tokenizer: Any) -> None:
    ss.init_tokenizer(tokenizer)


def jev_to_schema(ss: types.ModuleType, body: dict[str, Any]) -> dict[str, Any]:
    return ss.jev_schema(body)


def resolve_slots(
    ss: types.ModuleType, schema: dict[str, Any], *, thinking: bool = False
) -> tuple[list[int], list[dict[str, Any]]]:
    """Return (template token ids, slots) for a one-step canvas read."""
    head = [] if thinking else list(ss.SCAFFOLD)
    return ss.template_for(schema, head, "")


def soft_targets_for_gold(
    schema_q: dict[str, Any], gold: list[Any]
) -> list[float]:
    """Align challenge gold [kind, options, probs] to structured_server label order."""
    kind, options, probs = gold
    if kind == "noul":
        # structured_server labels are yes/no; gold.options are ("yes","no").
        return [float(p) for p in probs]
    if kind == "choice":
        # labels A,B,... map 1:1 to options order in criteria.
        by_name = dict(zip(options, probs, strict=True))
        names = [c[0] for c in schema_q["choices"]]
        return [float(by_name[n]) for n in names]
    if kind == "score":
        by_name = dict(zip(map(str, options), probs, strict=True))
        names = [c[0] for c in schema_q["choices"]]
        return [float(by_name[n]) for n in names]
    raise ValueError(f"unknown gold kind {kind!r}")


def slot_soft_ce(
    label_logprobs: list[float], target: list[float], eps: float = 1e-8
) -> tuple[float, float, bool]:
    """Soft CE on label tokens + label_mass. Returns (ce, label_mass, argmax_is_label).

    ``label_logprobs`` are raw log-probabilities (or logits) for each label id at the slot.
    Mirrors structured_server.slot_distribution: softmax over labels, mass = sum exp(lp).
    """
    import math

    mx = max(label_logprobs)
    ex = [math.exp(x - mx) for x in label_logprobs]
    z = sum(ex)
    probs = [e / z for e in ex]
    ce = -sum(t * math.log(max(p, eps)) for t, p in zip(target, probs, strict=True))
    # Approximate label_mass when logprobs are already absolute model logprobs.
    label_mass = sum(math.exp(x) for x in label_logprobs)
    argmax_is_label = True  # caller should set from full vocab argmax when available
    return ce, label_mass, argmax_is_label
