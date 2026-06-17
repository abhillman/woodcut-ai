"""Pluggable stylization adapter interface.

The 'stylize' stage produces the woodblock *look* from a source photo. It is the
slot you will benchmark across models/providers. Everything downstream
(separation, vectorization, laser) is deterministic CV and does not care which
adapter produced the image — so you can swap providers without touching the
pipeline.

Implement `stylize()` and register your adapter in `stylize/__init__.py`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StylizeAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def stylize(
        self,
        photo_path: Path,
        prompt: str,
        out_path: Path,
        *,
        n_colors: int = 5,
        negative_prompt: str = "",
    ) -> Path:
        """Render a stylized image from `photo_path` to `out_path`; return out_path."""
        raise NotImplementedError
