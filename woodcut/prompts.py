"""Killion-informed prompts.

Two audiences:
  * Claude (vision analysis + judging): reason about composition and carvability.
  * Diffusion models (stylization): produce the woodblock *look*.

These are the seed prompts the benchmark harness mutates into variants.
"""
from __future__ import annotations

# Shared aesthetic vocabulary, distilled from Killion's "faux ukiyo-e" lineage:
# Hokusai/Hiroshige landscape composition + Eric Gill / Rockwell Kent wood-
# engraving line. Flat color, bold silhouette, limited palette, no photoreal.
AESTHETIC = (
    "faux ukiyo-e woodblock print in the tradition of Tom Killion; lineage of "
    "Hokusai and Hiroshige landscape prints crossed with Eric Gill and Rockwell "
    "Kent wood-engraving; flat areas of color, bold confident outlines, strong "
    "silhouettes, limited harmonious palette, split-fountain sky gradients, "
    "no photorealism, no soft gradients within shapes, no fine texture noise"
)

# --- Claude: vision analysis -> BlockPlan ---------------------------------

ANALYSIS_SYSTEM = (
    "You are a master woodblock printmaker analyzing a landscape photograph the "
    "way Tom Killion analyzes his on-site sketches. Killion never prints from "
    "photos directly — he aggressively SIMPLIFIES a scene into a small number of "
    "carvable, flat-color blocks. Your job is the same reduction.\n\n"
    "Process facts to honor:\n"
    "- The KEY BLOCK is the darkest, most detailed block: outlines and visual "
    "anchors. It is printed LAST and every color block registers to it. Exactly "
    "one layer must be the key block.\n"
    "- Color blocks print LIGHTEST FIRST, darkest last (order 0 = lightest).\n"
    "- Skies are often a single split-fountain block (one block, color gradient).\n"
    "- Semi-transparent overlays let a few blocks yield many colors; use opacity "
    "<1.0 where overprinting should blend.\n"
    "- Favor 4–6 layers total. More blocks = more carving and registration risk.\n"
    "- Think in BOLD SHAPES a gouge can actually cut, not photographic detail."
)

ANALYSIS_USER = (
    "Analyze this photograph and produce a block plan. Identify the tonal/colour "
    "layers from foreground to sky, choose a palette in the {aesthetic}, decide "
    "which silhouette carries the key block, and note where flat areas should be "
    "simplified and where a split-fountain gradient fits. Default to "
    "{mode} production. Aim for {target_layers} layers."
).format(aesthetic=AESTHETIC, mode="{mode}", target_layers="{target_layers}")


def analysis_user_prompt(mode: str = "separate", target_layers: int = 5) -> str:
    return ANALYSIS_USER.format(mode=mode, target_layers=target_layers)


# --- Diffusion: stylization prompt ----------------------------------------

def stylize_prompt(subject: str, palette_hint: str = "", n_colors: int = 5) -> str:
    """Build a diffusion prompt for the stylization slot."""
    palette = f" Palette: {palette_hint}." if palette_hint else ""
    return (
        f"{AESTHETIC}. Subject: {subject}. Reduce to about {n_colors} flat colors."
        f"{palette} Clean carvable shapes, strong outlines, print-ready."
    )


STYLIZE_NEGATIVE = (
    "photorealistic, photograph, 3d render, soft focus, gradient shading inside "
    "shapes, busy texture, watermark, text, blur, noise"
)


# --- Claude: LLM-judge ----------------------------------------------------

JUDGE_SYSTEM = (
    "You are judging candidate woodblock-print reductions of a source photo for a "
    "printmaker working in Tom Killion's faux-ukiyo-e style who will LASER-CUT the "
    "blocks. Score honestly and comparatively. A beautiful image that cannot be "
    "cleanly carved into a few registered blocks is a bad result."
)

# Scored 0-10 each; the harness aggregates.
JUDGE_CRITERIA = [
    ("woodblock_fidelity", "Reads as a flat-color woodblock print, not a filtered photo."),
    ("shape_boldness", "Shapes are bold, confident, and gouge-carvable."),
    ("layer_separation", "Colors separate cleanly into distinct, printable blocks."),
    ("carvability", "Few enough blocks; no slivers/islands that would crumble in wood."),
    ("killion_resemblance", "Evokes Killion / Hokusai / Hiroshige landscape sensibility."),
]
