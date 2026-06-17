"""Color separation — deterministic CV, no generative AI.

Tuned for a *sparse* ukiyo-e / Killion look: mostly bare paper, a few flat ink
areas, and confident outlines — not a recolored photo. Pipeline:

  1. FLATTEN hard into big coherent regions (mode-filter at capped resolution).
  2. k-means into the planned color slots; map each cluster to its nearest ink.
  3. Emit one mask per PRINTED color layer; layers marked `is_background` are
     left as bare paper (negative space) and produce no block.
  4. KEY BLOCK = the boundaries between the flat regions (the seams), dilated to
     ink weight and despeckled — a few decisive lines, not edge-detection hatch.
  5. Despeckle every mask: small islands/slivers drop to bare paper (carvable).

Masks are 1-bit PNGs (white = printed area). These feed `vectorize.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage  # available via scikit-learn's dependency
from sklearn.cluster import KMeans

from .models import BlockPlan, RasterLayer

WORK_DIM = 900          # cap working resolution (strong + fast flattening)
MIN_REGION_FRAC = 0.0015  # drop connected regions smaller than this fraction of area
LINE_WEIGHT = 2         # key-block outline thickness in px


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 else n + 1


def flatten(img: Image.Image, strength: int = 0, passes: int = 2) -> Image.Image:
    """Collapse detail into big flat regions with a mode filter.

    `strength` is the filter window in px; 0 auto-scales to the image. Larger =
    flatter/sparser. Run at capped resolution by the caller for speed.
    """
    rgb = img.convert("RGB")
    if strength <= 0:
        strength = _odd(max(5, min(rgb.size) // 90))
    else:
        strength = _odd(strength)
    for _ in range(passes):
        rgb = rgb.filter(ImageFilter.ModeFilter(size=strength))
    return rgb


def posterize(img: Image.Image, n_colors: int = 5, seed: int = 0) -> Image.Image:
    """Flatten an image to `n_colors` flat regions via k-means (returns RGB)."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    km = KMeans(n_clusters=n_colors, n_init=4, random_state=seed)
    labels = km.fit_predict(rgb.reshape(-1, 3))
    centers = km.cluster_centers_.astype(np.uint8)
    return Image.fromarray(centers[labels].reshape(h, w, 3), "RGB")


def _clean(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Open/close to smooth, then drop connected components below `min_area`."""
    m = ndimage.binary_opening(mask, iterations=1)
    m = ndimage.binary_closing(m, iterations=2)
    labeled, n = ndimage.label(m)
    if n:
        counts = np.bincount(labeled.ravel())
        small = np.nonzero(counts < min_area)[0]
        small = small[small != 0]
        if small.size:
            m[np.isin(labeled, small)] = False
    return m


def _region_boundaries(labels: np.ndarray) -> np.ndarray:
    """Pixels lying on a seam between two flat regions (clean outlines)."""
    b = np.zeros(labels.shape, dtype=bool)
    diff_h = labels[:, :-1] != labels[:, 1:]
    diff_v = labels[:-1, :] != labels[1:, :]
    b[:, :-1] |= diff_h
    b[:, 1:] |= diff_h
    b[:-1, :] |= diff_v
    b[1:, :] |= diff_v
    return b


def separate_layers(
    image: Image.Image,
    plan: BlockPlan,
    out_dir: Path,
    seed: int = 0,
    *,
    simplify: int = 0,
    min_region_frac: float = MIN_REGION_FRAC,
    line_weight: int = LINE_WEIGHT,
) -> list[RasterLayer]:
    """Produce one mask per PRINTED layer in `plan`. Returns RasterLayer descriptors."""
    out_dir.mkdir(parents=True, exist_ok=True)
    color_layers = plan.color_layers_print_order()   # printed + background, excl. key
    key = plan.key_block()

    # Work at capped resolution so flattening is strong and fast.
    work = image.convert("RGB")
    if max(work.size) > WORK_DIM:
        work.thumbnail((WORK_DIM, WORK_DIM))
    flat_img = flatten(work, strength=simplify)

    rgb = np.asarray(flat_img, dtype=np.float32)
    h, w, _ = rgb.shape
    min_area = max(16, int(min_region_frac * h * w))

    # One cluster per color slot (printed + background); assign to nearest ink.
    n = max(1, len(color_layers))
    km = KMeans(n_clusters=n, n_init=4, random_state=seed)
    labels = km.fit_predict(rgb.reshape(-1, 3)).reshape(h, w)
    centers = km.cluster_centers_

    plan_colors = np.array([_hex_to_rgb(l.hex_color) for l in color_layers], dtype=np.float32)
    cluster_to_layer = {
        c: int(np.argmin(np.linalg.norm(plan_colors - centers[c], axis=1)))
        for c in range(n)
    }

    results: list[RasterLayer] = []
    for li, layer in enumerate(color_layers):
        if getattr(layer, "is_background", False):
            continue  # negative space — left as bare paper, no block
        member = [c for c, m in cluster_to_layer.items() if m == li]
        mask = _clean(np.isin(labels, member), min_area)
        if mask.any():
            results.append(_save_mask(mask, layer.name, layer.order, False,
                                      layer.hex_color, layer.opacity, out_dir))

    # Key block: seams between the flat regions, inked to weight, despeckled.
    boundaries = _region_boundaries(labels)
    if line_weight > 1:
        boundaries = ndimage.binary_dilation(boundaries, iterations=line_weight - 1)
    boundaries = _clean(boundaries, max(8, min_area // 4))
    results.append(_save_mask(boundaries, key.name, key.order, True,
                              key.hex_color, key.opacity, out_dir))
    return results


def _save_mask(
    mask: np.ndarray, name: str, order: int, is_key: bool,
    hex_color: str, opacity: float, out_dir: Path
) -> RasterLayer:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") or f"layer{order}"
    fname = f"{order:02d}_{'key_' if is_key else ''}{safe}.png"
    path = out_dir / fname
    Image.fromarray((mask * 255).astype(np.uint8), "L").save(path)
    return RasterLayer(
        layer_name=name,
        order=order,
        is_key_block=is_key,
        mask_path=str(path),
        hex_color=hex_color,
        opacity=opacity,
    )
