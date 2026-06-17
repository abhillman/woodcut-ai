"""Build a contact sheet of candidate previews with their judge scores.

Taste isn't fully automatable — the contact sheet is where YOU make the final
call. Rows = photos, columns = matrix cells; each tile shows the stylized image
(or a mask preview) captioned with the cell label and Claude's overall score.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

TILE = 256
PAD = 8
CAPTION_H = 30


def _load_preview(rec: dict) -> Image.Image:
    """Prefer the stylized PNG; else fall back to the key-block mask if present."""
    candidate = rec.get("stylized")
    if candidate and Path(candidate).exists() and candidate.endswith((".png", ".jpg", ".jpeg")):
        return Image.open(candidate).convert("RGB")
    masks = Path(rec["out_dir"]) / "masks"
    if masks.exists():
        keys = sorted(masks.glob("*key*.png"))
        if keys:
            return Image.open(keys[0]).convert("RGB")
    return Image.new("RGB", (TILE, TILE), "#dddddd")


def build_contact_sheet(results: list[dict], out_path: Path) -> Path:
    photos = sorted({r["photo"] for r in results})
    cells = sorted({r["label"] for r in results})
    by_key = {(r["photo"], r["label"]): r for r in results}

    cols, rows = len(cells), len(photos)
    cell_h = TILE + CAPTION_H
    W = PAD + cols * (TILE + PAD)
    H = PAD + rows * (cell_h + PAD)
    sheet = Image.new("RGB", (W, H), "#f3efe6")
    draw = ImageDraw.Draw(sheet)

    for r, photo in enumerate(photos):
        for c, label in enumerate(cells):
            rec = by_key.get((photo, label))
            x = PAD + c * (TILE + PAD)
            y = PAD + r * (cell_h + PAD)
            if rec:
                img = _load_preview(rec).resize((TILE, TILE))
                sheet.paste(img, (x, y))
                score = rec.get("overall")
                cap = f"{label}  {score:.1f}" if score is not None else label
                draw.rectangle([x, y + TILE, x + TILE, y + cell_h], fill="#222")
                draw.text((x + 4, y + TILE + 8), f"{photo}\n{cap}"[:60], fill="#fff")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path
