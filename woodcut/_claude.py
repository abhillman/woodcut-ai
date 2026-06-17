"""Thin Claude helpers shared by analysis + judge."""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import anthropic


@lru_cache
def client() -> anthropic.Anthropic:
    # Resolves ANTHROPIC_API_KEY from the environment.
    return anthropic.Anthropic()


def image_block(path: str | Path) -> dict:
    """Build a vision content block from an image file (base64)."""
    p = Path(path)
    ext = p.suffix.lower().lstrip(".")
    media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
             "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/jpeg")
    data = base64.standard_b64encode(p.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media, "data": data},
    }
