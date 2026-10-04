"""Local pins that must stay consistent with opentype_challenge.pins."""

from __future__ import annotations

import hashlib
from pathlib import Path

from opentype_challenge import pins as challenge_pins

from .paths import STRUCTURED_SERVER

BASE_REPO = challenge_pins.BASE_REPO
BASE_REVISION = challenge_pins.BASE_REVISION
BASE_CONFIG_SHA256 = challenge_pins.BASE_FILES["config.json"]
STRUCTURED_SERVER_URL = challenge_pins.STRUCTURED_SERVER_URL
STRUCTURED_SERVER_SHA256 = challenge_pins.STRUCTURED_SERVER_SHA256
VLLM_IMAGE = challenge_pins.VLLM_IMAGE
CANVAS = 256
MAX_MODEL_LEN = 131072
VOCAB = 262144
CHALLENGE_VERSION = "2.2.0"

# Default LoRA targets: attn + dense MLP; router frozen; vision separate.
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_structured_server(path: Path = STRUCTURED_SERVER) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run: opentype-miner recon --fetch-structured-server"
        )
    actual = sha256_file(path)
    if actual != STRUCTURED_SERVER_SHA256:
        raise RuntimeError(
            f"{path} sha256 {actual} != pinned {STRUCTURED_SERVER_SHA256}"
        )
    return path
