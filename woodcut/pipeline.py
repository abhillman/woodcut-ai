"""End-to-end orchestration: photo -> stylize -> separate -> vectorize -> laser.

This is the hybrid path:
  1. Claude analyzes the photo into a BlockPlan (or a heuristic plan offline).
  2. The stylize adapter renders the woodblock 'look' (stub by default).
  3. Classical CV separates the stylized image into per-block masks.
  4. potrace vectorizes each mask; we assemble per-block + preview SVGs.

Returns a PrintProject describing every artifact written under `out_dir`.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .analysis import analyze_photo
from .config import Config, load_config
from .laser import write_block_svgs, write_preview_png
from .models import PrintProject, ProductionMode
from .prompts import STYLIZE_NEGATIVE, stylize_prompt
from .separate import separate_layers
from .sheet import build_block_sheet
from .stylize import get_adapter


def run_pipeline(
    photo_path: str | Path,
    out_dir: str | Path,
    cfg: Config | None = None,
    *,
    mode: ProductionMode = ProductionMode.SEPARATE,
    target_layers: int = 5,
    use_stylizer: bool = True,
) -> PrintProject:
    cfg = cfg or load_config()
    photo_path = Path(photo_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Analyze -> plan
    plan = analyze_photo(photo_path, cfg, mode=mode, target_layers=target_layers)
    (out_dir / "plan.json").write_text(plan.model_dump_json(indent=2))

    # 2. Stylize (optional). Source for separation is stylized image or the photo.
    stylized_path: str | None = None
    sep_source = photo_path
    if use_stylizer:
        adapter = get_adapter(cfg.stylize_adapter)
        out = out_dir / "stylized.png"
        prompt = stylize_prompt(
            subject=plan.title,
            palette_hint=", ".join(l.hex_color for l in plan.layers),
            n_colors=len(plan.layers),
        )
        adapter.stylize(photo_path, prompt, out,
                        n_colors=len(plan.layers), negative_prompt=STYLIZE_NEGATIVE)
        stylized_path = str(out)
        sep_source = out

    # 3. Separate into per-block masks
    image = Image.open(sep_source).convert("RGB")
    raster_layers = separate_layers(image, plan, out_dir / "masks")

    project = PrintProject(
        source_photo=str(photo_path),
        plan=plan,
        stylized_path=stylized_path,
        raster_layers=raster_layers,
    )

    # 4. Vectorize + assemble laser files, plus preview + per-block sheet
    write_block_svgs(project, out_dir / "blocks")
    project.preview_path = str(write_preview_png(project, out_dir / "preview.png"))
    project.block_sheet_path = str(build_block_sheet(project, out_dir / "block_sheet.png"))

    (out_dir / "project.json").write_text(project.model_dump_json(indent=2))
    return project
