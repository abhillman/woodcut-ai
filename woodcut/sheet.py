"""Labeled per-block sheet for a single print.

Renders each block on its own tile — the carved shape shown in its ink color on
the paper ground — captioned with print order, name, key/color role, hex, and
opacity. This is the "here's what each block looks like" reference, distinct
from the benchmark contact sheet (which compares whole candidates).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .models import PrintProject
from .separate import _hex_to_rgb

TILE = 240
CAPTION_H = 52
PAD = 12
COLS = 4
PAPER = "#f3efe6"


def _font(size: int = 13) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _render_block_tile(mask_path: str, hex_color: str) -> Image.Image:
    """Ink-tinted block shape (full opacity, for clarity) on the paper ground."""
    with Image.open(mask_path) as m:
        mask = m.convert("L")
    paper = Image.new("RGB", mask.size, _hex_to_rgb(PAPER))
    ink = Image.new("RGB", mask.size, _hex_to_rgb(hex_color))
    composed = Image.composite(ink, paper, mask)  # white in mask -> ink
    composed.thumbnail((TILE, TILE))
    tile = Image.new("RGB", (TILE, TILE), _hex_to_rgb(PAPER))
    tile.paste(composed, ((TILE - composed.width) // 2, (TILE - composed.height) // 2))
    return tile


def build_block_sheet(project: PrintProject, out_path: Path) -> Path:
    layers = sorted(project.raster_layers, key=lambda l: l.order)
    n = len(layers)
    cols = min(COLS, n) or 1
    rows = (n + cols - 1) // cols
    cell_h = TILE + CAPTION_H
    W = PAD + cols * (TILE + PAD)
    H = PAD + rows * (cell_h + PAD) + 28  # +title strip

    sheet = Image.new("RGB", (W, H), "#ffffff")
    draw = ImageDraw.Draw(sheet)
    title = f"{project.plan.title}  ·  {project.plan.mode.value}  ·  {n} blocks"
    draw.text((PAD, 6), title, fill="#222", font=_font(15))

    for i, layer in enumerate(layers):
        r, c = divmod(i, cols)
        x = PAD + c * (TILE + PAD)
        y = 28 + PAD + r * (cell_h + PAD)
        sheet.paste(_render_block_tile(layer.mask_path, layer.hex_color), (x, y))

        role = "KEY (prints last)" if layer.is_key_block else "color"
        draw.rectangle([x, y + TILE, x + TILE, y + cell_h], fill="#222")
        line1 = f"{layer.order:02d}  {layer.layer_name[:22]}"
        line2 = f"{role}  {layer.hex_color}  op {layer.opacity:.2f}"
        draw.text((x + 6, y + TILE + 6), line1, fill="#fff", font=_font(13))
        draw.text((x + 6, y + TILE + 26), line2, fill="#cdb", font=_font(12))
        # swatch chip
        draw.rectangle([x + TILE - 26, y + TILE + 8, x + TILE - 8, y + TILE + 26],
                       fill=_hex_to_rgb(layer.hex_color), outline="#fff")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path
