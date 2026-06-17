"""Run the experiment matrix over a folder of photos and judge each result.

Writes, under `benchmark_runs/<timestamp>/`:
  * one output dir per (photo, cell) with all artifacts
  * results.json  -- every cell's judge scores
  * contact_sheet.png -- side-by-side previews + scores for human taste-calls

The whole thing runs offline (heuristic plan + stub stylizer + no judge); set
ANTHROPIC_API_KEY to get real Claude analysis and scoring.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

from woodcut.config import load_config
from woodcut.judge import judge_result
from woodcut.pipeline import run_pipeline

from .contact_sheet import build_contact_sheet
from .matrix import Cell, default_matrix

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def find_photos(photos_dir: Path) -> list[Path]:
    return sorted(p for p in photos_dir.iterdir() if p.suffix.lower() in PHOTO_EXTS)


def run_benchmark(
    photos_dir: str | Path = "photos",
    cells: list[Cell] | None = None,
    judge: bool = True,
) -> Path:
    cfg = load_config()
    cells = cells or default_matrix()
    photos = find_photos(Path(photos_dir))
    if not photos:
        raise SystemExit(f"No photos found in {photos_dir}/ (add .jpg/.png files).")

    run_dir = Path("benchmark_runs") / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for photo in photos:
        for cell in cells:
            out_dir = run_dir / photo.stem / cell.label
            project = run_pipeline(
                photo, out_dir, cfg,
                mode=cell.mode,
                target_layers=cell.target_layers,
                use_stylizer=cell.use_stylizer,
            )
            # Judge the most representative artifact we have (stylized > preview).
            candidate = project.stylized_path or project.svg_path
            score = None
            if judge and candidate and candidate.endswith((".png", ".jpg", ".jpeg")):
                score = judge_result(photo, candidate, cfg, label=project.plan.title)

            results.append({
                "photo": photo.name,
                "cell": dataclasses.asdict(cell) | {"mode": cell.mode.value},
                "label": cell.label,
                "out_dir": str(out_dir),
                "stylized": project.stylized_path,
                "preview": project.svg_path,
                "overall": score.overall if score else None,
                "verdict": score.verdict if score else None,
            })
            print(f"  {photo.name} :: {cell.label} "
                  f"-> {results[-1]['overall'] if score else 'no-judge'}")

    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    sheet = build_contact_sheet(results, run_dir / "contact_sheet.png")
    print(f"\nResults: {run_dir/'results.json'}\nContact sheet: {sheet}")
    return run_dir
