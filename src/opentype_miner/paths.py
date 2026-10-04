"""Repo-relative paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor"
STRUCTURED_SERVER = VENDOR / "structured_server.py"
DATA = ROOT / "data"
PUBLIC = DATA / "public"
CLOSED = DATA / "closed"
SYNTH = DATA / "synth"
ON_POLICY = DATA / "on_policy"
PRIVATE_PROXY = DATA / "private_proxy"
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
SUBMIT = ROOT / "submit"
RECON = ROOT / "RECON.md"
MIX_YAML = DATA / "mix.yaml"
