"""Replicate stylization adapter (img2img).

Runs the photo through a hosted diffusion model on Replicate to produce the
woodblock 'look', preserving composition via img2img `prompt_strength`. The
model slug is configurable so the benchmark can sweep SDXL / Flux / ControlNet
variants through this one adapter.

Env:
  REPLICATE_API_TOKEN   (required for real calls; the replicate client reads it)
  REPLICATE_MODEL       model slug, default 'stability-ai/sdxl'
  WOODCUT_STYLIZE_STRENGTH   img2img strength 0..1 (lower = closer to photo)

The network call is isolated in `_run()` so tests can monkeypatch it.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from .base import StylizeAdapter

DEFAULT_MODEL = "stability-ai/sdxl"


class ReplicateAdapter(StylizeAdapter):
    name = "replicate"

    def __init__(self) -> None:
        self.model = os.environ.get("REPLICATE_MODEL", DEFAULT_MODEL)
        self.strength = float(os.environ.get("WOODCUT_STYLIZE_STRENGTH", "0.65"))

    def stylize(
        self,
        photo_path: Path,
        prompt: str,
        out_path: Path,
        *,
        n_colors: int = 5,
        negative_prompt: str = "",
    ) -> Path:
        item = self._run(photo_path, prompt, negative_prompt)
        data = _fetch_bytes(item)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        return out_path

    def _run(self, photo_path: Path, prompt: str, negative_prompt: str):
        """Execute the Replicate prediction; return the first output item."""
        try:
            import replicate
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "The 'replicate' package is required for the replicate adapter. "
                "Install it: pip install 'woodcut-ai[providers]'"
            ) from e
        if not os.environ.get("REPLICATE_API_TOKEN"):
            raise RuntimeError("REPLICATE_API_TOKEN is not set.")

        with open(photo_path, "rb") as img:
            output = replicate.run(
                self.model,
                input={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "image": img,
                    "prompt_strength": self.strength,
                    "num_outputs": 1,
                },
            )
        # Output may be a single item or a list of FileOutput/str.
        return output[0] if isinstance(output, (list, tuple)) else output


def _fetch_bytes(item) -> bytes:
    """Turn a Replicate output item (FileOutput | URL str) into image bytes."""
    if hasattr(item, "read"):          # FileOutput (replicate>=0.25)
        return item.read()
    url = getattr(item, "url", None) or (item if isinstance(item, str) else None)
    if not url:
        raise RuntimeError(f"Unrecognized Replicate output: {item!r}")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (trusted provider URL)
        return resp.read()
