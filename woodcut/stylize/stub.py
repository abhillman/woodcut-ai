"""No-network stub adapter.

Stands in for a diffusion provider so the full pipeline + benchmark run today.
It approximates the woodblock 'look' with deterministic CV: bilateral-style
smoothing (via PIL) + posterization to `n_colors`. The output is intentionally
unglamorous — it's a placeholder whose job is to exercise the plumbing, not to
compete with a real diffusion model.

To wire a real provider, copy this file, implement `stylize()` against your API
or local endpoint, and register it in `__init__.py`.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter

from .base import StylizeAdapter
from ..separate import posterize  # reuse the quantizer


class StubStylizeAdapter(StylizeAdapter):
    name = "stub"

    def stylize(
        self,
        photo_path: Path,
        prompt: str,
        out_path: Path,
        *,
        n_colors: int = 5,
        negative_prompt: str = "",
    ) -> Path:
        img = Image.open(photo_path).convert("RGB")
        # Smooth detail away (poor man's "flatten into shapes"), then posterize.
        smoothed = img.filter(ImageFilter.MedianFilter(size=5))
        flat = posterize(smoothed, n_colors=n_colors)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        flat.save(out_path)
        return out_path
