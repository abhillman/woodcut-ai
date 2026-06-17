"""Domain model for a multi-block woodblock print.

Modeled directly on Tom Killion's described process:

  * The KEY BLOCK is carved first. It is the darkest, most detailed block and
    carries the outlines plus the registration information every color block
    aligns to.
  * COLOR BLOCKS are derived from the key block and printed lightest -> darkest,
    with the key block printed LAST.
  * Two production modes:
      - "separate"  : one physical block per color.
      - "reduction" : a single block carved progressively, overprinting earlier
                      lighter colors with later darker ones (block is destroyed
                      as it goes; layer ORDER is load-bearing).
  * Color blocks may overprint with semi-transparent inks, and skies are often a
    single block inked split-fountain (a gradient across one block).

These are pydantic models so Claude's structured-output analysis can populate
them directly via `client.messages.parse(...)`.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ProductionMode(str, Enum):
    SEPARATE = "separate"     # one block per color
    REDUCTION = "reduction"   # single block, carved progressively


class Layer(BaseModel):
    """One ink pass = one color, printed in `order` (0 = lightest, first down)."""
    name: str = Field(description="Human label, e.g. 'sky gradient', 'granite midtone'.")
    order: int = Field(description="Print order; 0 prints first (lightest).")
    hex_color: str = Field(description="Target ink color, e.g. '#9fb6c4'.")
    opacity: float = Field(default=1.0, ge=0.0, le=1.0,
                           description="Ink transparency; <1 means it overprints/blends.")
    split_fountain: bool = Field(
        default=False,
        description="True for a single-block gradient (e.g. dawn sky).",
    )
    is_key_block: bool = Field(
        default=False,
        description="The darkest outline/detail block. Printed LAST. Exactly one per print.",
    )
    is_background: bool = Field(
        default=False,
        description="Negative space: left as BARE PAPER, no block carved or printed. "
                    "Use generously — unprinted paper is the main design element in "
                    "ukiyo-e/Killion prints. Typically the lightest region.",
    )
    notes: str = Field(default="", description="Carving guidance from the analysis.")


class BlockPlan(BaseModel):
    """Claude's reduction of a photo into a carvable, printable plan."""
    title: str
    mode: ProductionMode = ProductionMode.SEPARATE
    palette_rationale: str = Field(
        description="Why this palette/number of layers suits the image in the Killion idiom."
    )
    composition_notes: str = Field(
        description="How to simplify the photo into bold shapes; what to foreground/cut."
    )
    layers: list[Layer] = Field(description="Ordered, lightest-first. Exactly one is_key_block=True.")

    def key_block(self) -> Layer:
        keys = [l for l in self.layers if l.is_key_block]
        if len(keys) != 1:
            raise ValueError(f"A plan must have exactly one key block, found {len(keys)}.")
        return keys[0]

    def color_layers_print_order(self) -> list[Layer]:
        """Color blocks lightest-first; key block excluded (it prints last separately)."""
        colors = [l for l in self.layers if not l.is_key_block]
        return sorted(colors, key=lambda l: l.order)


# ---- Output artifacts (produced by the CV stages, not by Claude) ----

class RasterLayer(BaseModel):
    """A separated raster mask for one Layer (path to a 1-bit / grayscale PNG)."""
    layer_name: str
    order: int
    is_key_block: bool
    mask_path: str
    hex_color: str
    opacity: float = 1.0

    model_config = {"arbitrary_types_allowed": True}


class PrintProject(BaseModel):
    """Everything needed to cut and print one design."""
    source_photo: str
    plan: BlockPlan
    stylized_path: str | None = None      # diffusion output, if used
    raster_layers: list[RasterLayer] = []
    preview_path: str | None = None       # flattened color mockup (PNG)
    block_sheet_path: str | None = None   # labeled per-block reference sheet (PNG)
    registration: Literal["corner_ticks", "kento", "none"] = "corner_ticks"
