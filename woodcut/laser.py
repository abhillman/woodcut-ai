"""Assemble laser-cuttable SVGs from vectorized layers.

Outputs, per project:
  * one SVG PER BLOCK (the file you actually send to the laser), each carrying
    identical registration marks so the carved blocks print in register — this
    is the digital equivalent of Killion pulling proof sheets off the key block
    to align every color block.
  * a combined color PREVIEW SVG (all layers stacked with their inks/opacity),
    printed lightest-first with the key block on top, to check the design.

Registration default: corner ticks (works on any laser/SVG importer). A 'kento'
(traditional L-corner + bar) option is stubbed for later.
"""
from __future__ import annotations

from pathlib import Path

import svgwrite
from PIL import Image

from .models import PrintProject, RasterLayer
from .separate import _hex_to_rgb
from .vectorize import potrace_available, trace_to_svg_file

REG_TICK = 24  # px registration tick length
MARGIN = 40
PAPER = "#f3efe6"  # warm Japanese-paper ground for the preview


def _dims(layers: list[RasterLayer]) -> tuple[int, int]:
    if not layers:
        return (800, 600)
    with Image.open(layers[0].mask_path) as im:
        return im.size


def _reg_marks_markup(w: int, h: int) -> str:
    """Identical corner ticks on every block => alignment in printing.

    Returned as an SVG fragment in the block's pixel coordinate space (the same
    viewBox we set on every block), so the marks land in the same place on each.
    """
    color = "#ff00ff"  # magenta: ignore on the laser, or give it its own pass
    lines = []
    for (x, y) in [(MARGIN, MARGIN), (w - MARGIN, MARGIN),
                   (MARGIN, h - MARGIN), (w - MARGIN, h - MARGIN)]:
        lines.append(f'<line x1="{x-REG_TICK}" y1="{y}" x2="{x+REG_TICK}" y2="{y}" '
                     f'stroke="{color}" stroke-width="1"/>')
        lines.append(f'<line x1="{x}" y1="{y-REG_TICK}" x2="{x}" y2="{y+REG_TICK}" '
                     f'stroke="{color}" stroke-width="1"/>')
    return (f'<g id="registration" transform="scale(1)">'
            + "".join(lines) + "</g>")


def write_block_svgs(project: PrintProject, out_dir: Path) -> list[Path]:
    """One SVG per block, with registration marks. Returns written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted(project.raster_layers, key=lambda l: l.order)
    w, h = _dims(layers)
    written: list[Path] = []
    traced_ok = potrace_available()
    reg = _reg_marks_markup(w, h)

    for layer in layers:
        path = out_dir / f"block_{layer.order:02d}_{'key' if layer.is_key_block else 'color'}.svg"
        if traced_ok and trace_to_svg_file(layer.mask_path, path):
            # potrace wrote a full SVG; inject registration marks before </svg>.
            # potrace scales the bitmap to a pt canvas, so wrap the marks in a
            # group scaled from our pixel space to potrace's via a viewBox match.
            _inject_before_close(path, reg, src_w=w, src_h=h)
        else:
            # Fallback: a self-contained SVG embedding the mask as a base64 data
            # URI (no file:// reference — opens anywhere) + reg marks.
            import base64
            data = base64.b64encode(Path(layer.mask_path).read_bytes()).decode("ascii")
            dwg = svgwrite.Drawing(str(path), size=(w, h), viewBox=f"0 0 {w} {h}")
            dwg.add(dwg.rect((0, 0), (w, h), fill="white"))
            dwg.add(dwg.image(href=f"data:image/png;base64,{data}",
                              insert=(0, 0), size=(w, h)))
            dwg.save()
            _inject_before_close(path, reg, src_w=w, src_h=h, has_viewbox=True)
        written.append(path)

    if not traced_ok:
        print("  [warn] potrace not found — block SVGs embed raster masks instead "
              "of vector paths. Install potrace for true cut paths.")
    return written


def _inject_before_close(path: Path, fragment: str, *, src_w: int, src_h: int,
                         has_viewbox: bool = False) -> None:
    """Insert an SVG fragment (in src pixel space) just before </svg>.

    If the document declares a viewBox in pixel space the fragment drops in as-is;
    otherwise (potrace's pt canvas) we wrap it so the reg marks map onto the same
    relative positions via the document's own width/height.
    """
    text = path.read_text()
    frag = fragment
    if not has_viewbox:
        import re
        m = re.search(r'<svg[^>]*\bwidth="([\d.]+)[^"]*"[^>]*\bheight="([\d.]+)', text)
        if m:
            doc_w, doc_h = float(m.group(1)), float(m.group(2))
            sx, sy = doc_w / src_w, doc_h / src_h
            frag = f'<g transform="scale({sx},{sy})">{fragment}</g>'
    path.write_text(text.replace("</svg>", frag + "</svg>", 1))


def write_preview_png(project: PrintProject, out_path: Path) -> Path:
    """Flattened color mockup of the final print — always opens (it's a PNG).

    Composites each layer in PRESS ORDER (color blocks lightest-first, key block
    last) tinted with its ink color over a warm Japanese-paper ground, honoring
    each layer's opacity so semi-transparent overprints read correctly. This is
    a mockup of the printed result, not a laser file (those are the block SVGs).
    """
    import numpy as np

    layers = sorted(project.raster_layers, key=lambda l: l.order)
    w, h = _dims(layers)
    canvas = np.empty((h, w, 3), dtype=np.float32)
    canvas[:] = np.array(_hex_to_rgb(PAPER), dtype=np.float32)

    ordered = [l for l in layers if not l.is_key_block] + [l for l in layers if l.is_key_block]
    for layer in ordered:
        with Image.open(layer.mask_path) as m:
            mask = np.asarray(m.convert("L"), dtype=np.float32) / 255.0
        if mask.shape != (h, w):  # guard against size drift between masks
            mask = np.asarray(Image.fromarray((mask * 255).astype("uint8")).resize((w, h)),
                              dtype=np.float32) / 255.0
        ink = np.array(_hex_to_rgb(layer.hex_color), dtype=np.float32)
        alpha = (mask * layer.opacity)[..., None]
        canvas = canvas * (1.0 - alpha) + ink * alpha

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas.clip(0, 255).astype("uint8"), "RGB").save(out_path)
    return out_path
