"""Smoke tests that need only the challenge package (no torch)."""

from __future__ import annotations

import json
from pathlib import Path

from opentype_miner.data.generate import write_jsonl, iter_rows
from opentype_miner.data.oracle_chat import convert
from opentype_miner.data.hard_mine import mine
from opentype_miner.eval.sim_duel import sim_jsonl, headroom_table
from opentype_miner.pins import STRUCTURED_SERVER_SHA256, sha256_file
from opentype_miner.paths import STRUCTURED_SERVER, ROOT
from opentype_miner.serve.read_protocol import soft_targets_for_gold


def test_structured_server_pin():
    assert STRUCTURED_SERVER.exists()
    assert sha256_file(STRUCTURED_SERVER) == STRUCTURED_SERVER_SHA256


def test_soft_targets_noul_and_choice():
    noul_q = {"choices": [("yes", None), ("no", None)], "type": "noul"}
    assert soft_targets_for_gold(noul_q, ["noul", ["yes", "no"], [0.7, 0.3]]) == [0.7, 0.3]
    choice_q = {
        "choices": [("a", None), ("b", None), ("c", None)],
        "type": "choice",
    }
    assert soft_targets_for_gold(choice_q, ["choice", ["a", "b", "c"], [0.0, 1.0, 0.0]]) == [
        0.0,
        1.0,
        0.0,
    ]


def test_generate_oracle_sim_and_hard_mine(tmp_path: Path):
    decisions = tmp_path / "d.jsonl"
    ops = tmp_path / "ops.jsonl"
    n = write_jsonl(decisions, iter_rows("decisions", 2, 8, seed=1, family=None))
    assert n == 8
    n = write_jsonl(ops, iter_rows("ops", None, 4, seed=2, family=None))
    assert n == 4
    chat = tmp_path / "chat.jsonl"
    assert convert(ops, chat) >= 1

    mix = tmp_path / "mix.jsonl"
    with mix.open("w") as out:
        out.write(decisions.read_text())
        out.write(ops.read_text())

    scores = tmp_path / "scores.jsonl"
    verdict = sim_jsonl(
        mix,
        champion_skill="blur:1.4",
        challenger_skill="blur:1.05",
        scores_out=scores,
    )
    assert "g_lcb" in verdict and "headroom" in verdict
    table = headroom_table(verdict)
    assert "decisions" in table

    mined = tmp_path / "hard.jsonl"
    summary = mine(scores, mix, mined, mix_path=ROOT / "data" / "mix.yaml")
    assert mined.exists() and mined.stat().st_size > 0
    assert sum(summary.values()) >= 1


def test_mix_yaml_exists():
    text = (ROOT / "data" / "mix.yaml").read_text()
    assert "ship_gate" in text and "0.42" in text
