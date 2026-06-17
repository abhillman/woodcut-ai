"""Define the experiment matrix to sweep over your photo set.

A 'cell' is one combination of knobs. The runner executes every cell on every
photo, judges the output with Claude, and you compare via the contact sheet.
Edit `default_matrix()` to add stylize adapters, layer counts, or modes.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from woodcut.models import ProductionMode


@dataclass(frozen=True)
class Cell:
    stylize_adapter: str
    target_layers: int
    mode: ProductionMode
    use_stylizer: bool

    @property
    def label(self) -> str:
        sty = self.stylize_adapter if self.use_stylizer else "cv-only"
        return f"{sty}_L{self.target_layers}_{self.mode.value}"


def default_matrix() -> list[Cell]:
    cells: list[Cell] = []
    # Classical-CV-only baseline (no diffusion) at a couple of layer counts.
    for layers in (4, 5, 6):
        cells.append(Cell("stub", layers, ProductionMode.SEPARATE, use_stylizer=False))
    # Hybrid (stylize + CV) — add real adapters to the first tuple element below.
    for adapter, layers, mode in product(
        ["stub"], (4, 5), (ProductionMode.SEPARATE, ProductionMode.REDUCTION)
    ):
        cells.append(Cell(adapter, layers, mode, use_stylizer=True))
    return cells
