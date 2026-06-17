"""Color separation — deterministic CV, no generative AI.

Turns a (stylized) image + a BlockPlan into one raster mask per layer:
  * color blocks: k-means cluster the image, assign each cluster to the nearest
    planned ink color, emit a binary mask per color layer.
  * key block: an outline/detail mask derived from edges + the darkest regions
    (the key block is "the darkest, most detailed block" carrying the outlines).

Masks are 1-bit PNGs (white = printed/raised area). These feed `vectorize.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans

from .models import BlockPlan, RasterLayer


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def posterize(img: Image.Image, n_colors: int = 5, seed: int = 0) -> Image.Image:
    """Flatten an image to `n_colors` flat regions via k-means (returns RGB)."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    km = KMeans(n_clusters=n_colors, n_init=4, random_state=seed)
    labels = km.fit_predict(flat)
    centers = km.cluster_centers_.astype(np.uint8)
    out = centers[labels].reshape(h, w, 3)
    return Image.fromarray(out, "RGB")


def _key_block_mask(img: Image.Image, edge_thresh: int = 40) -> np.ndarray:
    """Outline + darkest-detail mask for the key block (bool array)."""
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.asarray(edges) > edge_thresh
    # Darkest ~20% of pixels also belong on the detail/key block.
    g = np.asarray(gray)
    dark = g < np.percentile(g, 20)
    return edge_arr | dark


def separate_layers(
    image: Image.Image,
    plan: BlockPlan,
    out_dir: Path,
    seed: int = 0,
) -> list[RasterLayer]:
    """Produce one mask per layer in `plan`. Returns RasterLayer descriptors."""
    out_dir.mkdir(parents=True, exist_ok=True)
    color_layers = plan.color_layers_print_order()
    key = plan.key_block()

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)

    # One cluster per color layer; assign clusters to nearest planned ink color.
    n = max(1, len(color_layers))
    km = KMeans(n_clusters=n, n_init=4, random_state=seed)
    labels = km.fit_predict(flat).reshape(h, w)
    centers = km.cluster_centers_

    plan_colors = np.array([_hex_to_rgb(l.hex_color) for l in color_layers], dtype=np.float32)
    # cluster index -> plan layer index (nearest color in RGB)
    cluster_to_layer = {
        c: int(np.argmin(np.linalg.norm(plan_colors - centers[c], axis=1)))
        for c in range(n)
    }

    results: list[RasterLayer] = []
    for li, layer in enumerate(color_layers):
        member_clusters = [c for c, l in cluster_to_layer.items() if l == li]
        mask = np.isin(labels, member_clusters)
        results.append(_save_mask(mask, layer.name, layer.order, False, layer.hex_color, out_dir))

    # Key block from edges + dark detail.
    key_mask = _key_block_mask(image)
    results.append(_save_mask(key_mask, key.name, key.order, True, key.hex_color, out_dir))
    return results


def _save_mask(
    mask: np.ndarray, name: str, order: int, is_key: bool, hex_color: str, out_dir: Path
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
    )
