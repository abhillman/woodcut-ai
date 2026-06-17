"""Smoke-test a diffusion stylization adapter against the real provider.

Runs one stylize() call through the chosen adapter and reports clear
diagnostics — token present? SDK installed? call succeeded? — so you can
confirm your tokens work before kicking off a full benchmark. Use a synthetic
swatch if you don't pass a photo, so it's a minimal/cheap request.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from PIL import Image, ImageDraw

from .prompts import STYLIZE_NEGATIVE, stylize_prompt
from .stylize import get_adapter

# Which env var holds each provider's credential.
_TOKEN_ENV = {
    "replicate": ["REPLICATE_API_TOKEN"],
    "fal": ["FAL_API_TOKEN", "FAL_KEY"],
    "stub": [],
}


def _swatch(path: Path) -> Path:
    """A tiny synthetic alpine scene so the smoke call is small and cheap."""
    img = Image.new("RGB", (512, 384), "#bcd3e0")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 512, 110], fill="#d8e4ec")
    d.polygon([(0, 180), (190, 80), (360, 150), (512, 110), (512, 220), (0, 220)], fill="#6f97a8")
    d.rectangle([0, 220, 512, 300], fill="#3f6b6e")
    d.rectangle([0, 300, 512, 384], fill="#23373a")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def run_smoke(adapter_name: str, photo: str | Path | None = None) -> tuple[bool, str, Path | None]:
    """Returns (ok, message, output_path). Never raises — diagnostics in message."""
    lines: list[str] = [f"adapter: {adapter_name}"]

    # Token check
    for env in _TOKEN_ENV.get(adapter_name, []):
        if os.environ.get(env):
            lines.append(f"token:   {env} is set ✓")
            break
    else:
        if _TOKEN_ENV.get(adapter_name):
            return False, "\n".join(lines + [
                f"token:   none of {_TOKEN_ENV[adapter_name]} are set ✗"]), None

    try:
        adapter = get_adapter(adapter_name)
    except KeyError as e:
        return False, "\n".join(lines + [f"error:   {e}"]), None
    lines.append(f"model:   {getattr(adapter, 'model', 'n/a')}")

    src = Path(photo) if photo else _swatch(Path("outputs/smoke/_swatch.png"))
    out = Path("outputs/smoke") / f"{adapter_name}.png"
    prompt = stylize_prompt(subject="Sierra Nevada alpine lake", n_colors=5)

    t0 = time.time()
    try:
        adapter.stylize(src, prompt, out, n_colors=5, negative_prompt=STYLIZE_NEGATIVE)
    except Exception as e:  # noqa: BLE001 - we want to surface any failure cleanly
        return False, "\n".join(lines + [f"call:    FAILED after {time.time()-t0:.1f}s",
                                          f"error:   {type(e).__name__}: {e}"]), None

    dt = time.time() - t0
    if not out.exists() or out.stat().st_size == 0:
        return False, "\n".join(lines + ["call:    returned but wrote no image ✗"]), None
    lines.append(f"call:    OK in {dt:.1f}s -> {out}")
    return True, "\n".join(lines), out
