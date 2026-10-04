"""Unit tests for block-diffusion helpers (requires torch)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from opentype_miner.train.diffusion_loss import (  # noqa: E402
    corrupt_uniform,
    diffusion_ce,
    soft_label_ce_from_logits,
)


def test_corrupt_and_ce():
    ids = torch.randint(0, 100, (2, 16))
    canvas = torch.zeros_like(ids, dtype=torch.bool)
    canvas[:, 8:] = True
    noisy, supervised, t = corrupt_uniform(ids, canvas, vocab_size=100, eps=0.001)
    assert noisy.shape == ids.shape
    assert supervised.shape == ids.shape
    assert t.shape == (2,)
    logits = torch.randn(2, 16, 100)
    loss = diffusion_ce(logits, ids, supervised)
    assert torch.isfinite(loss)


def test_soft_label_ce():
    logits = torch.randn(50)
    label_ids = torch.tensor([1, 2, 3])
    target = torch.tensor([0.2, 0.5, 0.3])
    ce, mass = soft_label_ce_from_logits(logits, label_ids, target)
    assert torch.isfinite(ce) and 0.0 < float(mass) <= 1.0 + 1e-5
