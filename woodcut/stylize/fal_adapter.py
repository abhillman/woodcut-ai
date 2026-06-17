"""fal.ai stylization adapter (img2img).

Same job as the Replicate adapter, on fal.ai's fast hosted diffusion. The photo
is sent inline as a downscaled data URI (no fal CDN upload), then img2img runs.
Model slug is configurable for benchmark sweeps.

Env:
  FAL_API_TOKEN   (required for real calls; bridged to FAL_KEY for the client)
  FAL_MODEL      model slug, default 'fal-ai/flux/dev/image-to-image'
  WOODCUT_STYLIZE_STRENGTH   img2img strength 0..1 (lower = closer to photo)

The network call is isolated in `_run()` so tests can monkeypatch it.
"""
from __future__ import annotations

import base64
import io
import os
import urllib.request
from pathlib import Path

from PIL import Image

from .base import StylizeAdapter

DEFAULT_MODEL = "fal-ai/flux/dev/image-to-image"
MAX_DIM = 1024  # downscale long edge before sending (cheaper img2img, small request)


def image_data_uri(photo_path: str | Path, max_dim: int = MAX_DIM) -> str:
    """Downscale + JPEG-encode a photo into a data: URI for fal's `image_url`.

    Passing the image inline avoids fal's CDN upload (and its separate
    storage-auth call) entirely — one less thing that can 403.
    """
    with Image.open(photo_path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class FalAdapter(StylizeAdapter):
    name = "fal"

    def __init__(self) -> None:
        self.model = os.environ.get("FAL_MODEL", DEFAULT_MODEL)
        self.strength = float(os.environ.get("WOODCUT_STYLIZE_STRENGTH", "0.65"))

    def stylize(
        self,
        photo_path: Path,
        prompt: str,
        out_path: Path,
        *,
        n_colors: int = 5,
        negative_prompt: str = "",
    ) -> Path:
        result = self._run(photo_path, prompt, negative_prompt)
        url = _first_image_url(result)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as resp:  # noqa: S310 (trusted provider URL)
            out_path.write_bytes(resp.read())
        return out_path

    def _run(self, photo_path: Path, prompt: str, negative_prompt: str) -> dict:
        """Run img2img with the image passed inline; return the fal result dict."""
        # fal's client reads FAL_KEY; the user sets FAL_API_TOKEN, so bridge it.
        token = os.environ.get("FAL_API_TOKEN") or os.environ.get("FAL_KEY")
        if token:
            os.environ.setdefault("FAL_KEY", token)
        elif not os.environ.get("FAL_KEY"):
            raise RuntimeError("FAL_API_TOKEN (or FAL_KEY) is not set.")

        try:
            import fal_client
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "The 'fal-client' package is required for the fal adapter. "
                "Install it: pip install 'woodcut-ai[providers]'"
            ) from e

        # Inline data URI (no fal CDN upload / storage-auth call).
        image_url = image_data_uri(photo_path)
        try:
            return fal_client.subscribe(
                self.model,
                arguments={
                    "image_url": image_url,
                    "prompt": prompt,
                    "strength": self.strength,
                    "num_images": 1,
                },
            )
        except Exception as e:  # noqa: BLE001 - re-wrap auth errors with guidance
            msg = str(e)
            if "403" in msg or "401" in msg or "Forbidden" in msg or "Unauthorized" in msg:
                raise RuntimeError(
                    "fal rejected the credential (auth error). Check that "
                    "FAL_API_TOKEN is a valid fal key in 'id:secret' form, has no "
                    "surrounding quotes/whitespace, and that billing is enabled on "
                    "your fal account. Original: " + msg
                ) from e
            raise


def _first_image_url(result: dict) -> str:
    images = (result or {}).get("images") or []
    if not images or "url" not in images[0]:
        raise RuntimeError(f"Unexpected fal result (no images[].url): {result!r}")
    return images[0]["url"]
