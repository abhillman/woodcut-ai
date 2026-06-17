"""Raster mask -> vector path. Shells out to `potrace` when available.

`potrace` produces clean Bezier outlines from a bitmap — exactly what a laser
needs. We feed it a PBM and get back an SVG `<path>`. If potrace isn't
installed, we degrade gracefully: a flag is set so the caller can still assemble
a (raster-backed) preview and warn the user, rather than crashing.

    install:  brew install potrace   |   apt-get install potrace
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


def potrace_available() -> bool:
    return shutil.which("potrace") is not None


def trace_to_svg_file(mask_path: str | Path, out_path: str | Path, turdsize: int = 8) -> bool:
    """Trace a 1-bit mask to a complete SVG document at `out_path`.

    `turdsize` drops speckles smaller than N pixels — critical for carvability
    (removes the slivers/islands that crumble in wood). Returns False (writing
    nothing) if potrace is unavailable, so the caller can fall back.

    Using potrace's own full-document output keeps the traced geometry and the
    canvas in the same coordinate space; the caller injects registration marks.
    """
    if not potrace_available():
        return False

    img = Image.open(mask_path).convert("1")
    with tempfile.TemporaryDirectory() as td:
        pbm = Path(td) / "in.pbm"
        img.save(pbm)
        subprocess.run(
            ["potrace", str(pbm), "-s", "-o", str(out_path),
             "--turdsize", str(turdsize), "--flat"],
            check=True,
            capture_output=True,
        )
    return True
