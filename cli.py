"""woodcut CLI.

    woodcut analyze  PHOTO                 # Claude -> block plan (JSON)
    woodcut make     PHOTO [-o OUT]        # full pipeline -> laser SVGs + preview
    woodcut bench    [--photos DIR]        # sweep the matrix + judge + contact sheet
    woodcut adapters                       # list registered stylize adapters

Runs fully offline (heuristic plan + stub stylizer); set ANTHROPIC_API_KEY for
real Claude analysis and judging.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from woodcut.config import load_config
from woodcut.models import ProductionMode
from woodcut.pipeline import run_pipeline
from woodcut.stylize import available_adapters


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="woodcut", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_an = sub.add_parser("analyze", help="Claude vision analysis -> block plan")
    p_an.add_argument("photo")
    p_an.add_argument("--layers", type=int, default=5)
    p_an.add_argument("--mode", choices=[m.value for m in ProductionMode], default="separate")

    p_mk = sub.add_parser("make", help="Full pipeline -> laser SVGs + preview")
    p_mk.add_argument("photo")
    p_mk.add_argument("-o", "--out", default=None)
    p_mk.add_argument("--layers", type=int, default=5)
    p_mk.add_argument("--mode", choices=[m.value for m in ProductionMode], default="separate")
    p_mk.add_argument("--no-stylize", action="store_true", help="classical-CV-only path")

    p_bench = sub.add_parser("bench", help="Sweep matrix over a photo folder")
    p_bench.add_argument("--photos", default="photos")
    p_bench.add_argument("--no-judge", action="store_true")

    sub.add_parser("adapters", help="List registered stylize adapters")

    args = parser.parse_args(argv)
    cfg = load_config()

    if args.cmd == "adapters":
        print("Registered stylize adapters:", ", ".join(available_adapters()))
        print("Active (WOODCUT_STYLIZE_ADAPTER):", cfg.stylize_adapter)
        return 0

    if args.cmd == "analyze":
        from woodcut.analysis import analyze_photo
        plan = analyze_photo(args.photo, cfg, mode=ProductionMode(args.mode),
                             target_layers=args.layers)
        print(plan.model_dump_json(indent=2))
        return 0

    if args.cmd == "make":
        out = Path(args.out or (Path("outputs") / Path(args.photo).stem))
        project = run_pipeline(
            args.photo, out, cfg,
            mode=ProductionMode(args.mode),
            target_layers=args.layers,
            use_stylizer=not args.no_stylize,
        )
        print(f"Wrote project to {out}/")
        print(f"  plan:    {out/'plan.json'}")
        print(f"  preview: {project.svg_path}")
        print(f"  blocks:  {out/'blocks'}/  ({len(project.raster_layers)} blocks)")
        if not cfg.claude_available:
            print("  [note] offline heuristic plan — set ANTHROPIC_API_KEY for Claude analysis.")
        return 0

    if args.cmd == "bench":
        from benchmark.run import run_benchmark
        run_benchmark(args.photos, judge=not args.no_judge)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
