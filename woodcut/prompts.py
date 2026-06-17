"""Prompts tuned for a SPARSE ukiyo-e / Killion woodblock look.

The north star is Hiroshige, Hokusai, and Tom Killion: mostly bare paper, a few
flat ink areas, and a handful of confident outlines. Negative space (the
unprinted paper, *ma*) is the primary design element — not a recolored photo.

Two audiences:
  * Claude (vision analysis + judging): plan a sparse, carvable reduction.
  * Diffusion models (stylization): produce the flat, sparse woodblock look.
"""
from __future__ import annotations

# Shared aesthetic vocabulary. Sparse, flat, negative-space-forward.
AESTHETIC = (
    "sparse Japanese woodblock print in the tradition of Hiroshige, Hokusai, and "
    "Tom Killion; mostly empty unprinted paper (generous negative space / 'ma'), "
    "a few large flat areas of color, a limited palette of 3-4 inks, confident "
    "minimal outlines, asymmetric composition, strong silhouettes; NO "
    "photorealism, NO shading or gradients inside shapes, NO fine texture, NO "
    "busy detail — radical simplification into bold flat shapes"
)

# --- Claude: vision analysis -> BlockPlan ---------------------------------

ANALYSIS_SYSTEM = (
    "You are a master woodblock printmaker reducing a photograph to a SPARSE "
    "print in the tradition of Hiroshige, Hokusai, and Tom Killion. The goal is "
    "radical simplification, not recoloring. These prints are mostly BARE PAPER "
    "with only a few flat ink shapes and a handful of decisive lines.\n\n"
    "Hard rules:\n"
    "- Use FEW layers: 2-4 PRINTED colors plus the key block. Fewer is better.\n"
    "- Mark at least one layer is_background=True — large areas left as BARE "
    "PAPER (negative space). The lightest/sky/empty regions should be paper, not "
    "ink. Be generous: empty paper is the whole point.\n"
    "- Exactly one layer is_key_block=True: the darkest, carrying only the few "
    "essential outlines/silhouettes. It is printed LAST. Keep it minimal — a few "
    "confident lines, never hatching or texture.\n"
    "- Color blocks print LIGHTEST FIRST (order 0 = lightest/first down).\n"
    "- Skies are often a single split-fountain block, or simply bare paper.\n"
    "- Think in a handful of BOLD FLAT SHAPES a gouge can cut, with lots of empty "
    "space around them. Eliminate detail aggressively."
)

ANALYSIS_USER = (
    "Reduce this photograph to a sparse block plan in the {aesthetic}. Decide the "
    "2-4 flat ink shapes worth keeping, mark the empty/lightest regions as "
    "is_background (bare paper), choose which silhouette carries the minimal key "
    "block, and pick a restrained palette. Prefer {target_layers} layers TOTAL "
    "(including key + background); use fewer if the image allows. Default to "
    "{mode} production."
).format(aesthetic=AESTHETIC, mode="{mode}", target_layers="{target_layers}")


def analysis_user_prompt(mode: str = "separate", target_layers: int = 4) -> str:
    return ANALYSIS_USER.format(mode=mode, target_layers=target_layers)


# --- Diffusion: stylization prompt ----------------------------------------

def stylize_prompt(subject: str, palette_hint: str = "", n_colors: int = 4) -> str:
    """Build a diffusion prompt for the stylization slot."""
    palette = f" Palette: {palette_hint}." if palette_hint else ""
    return (
        f"{AESTHETIC}. Subject: {subject}. Reduce to {max(2, n_colors)} flat inks "
        f"over large areas of empty paper.{palette} Bold, sparse, lots of "
        f"negative space, confident minimal linework, print-ready."
    )


STYLIZE_NEGATIVE = (
    "photorealistic, photograph, 3d render, soft focus, gradient shading inside "
    "shapes, busy texture, fine detail, hatching, cluttered, dense, watermark, "
    "text, blur, noise"
)


# --- Claude: LLM-judge ----------------------------------------------------

JUDGE_SYSTEM = (
    "You are judging candidate woodblock-print reductions of a source photo for a "
    "printmaker working in the SPARSE Hiroshige/Hokusai/Killion tradition who will "
    "LASER-CUT the blocks. Reward radical simplification and generous negative "
    "space. Penalize busy, dense, or photo-like results, and anything that "
    "couldn't be cut into a few registered blocks."
)

# Scored 0-10 each; the harness aggregates.
JUDGE_CRITERIA = [
    ("negative_space", "Generous bare paper; sparse, uncluttered, breathing room."),
    ("shape_boldness", "A few bold, confident, flat, gouge-carvable shapes."),
    ("woodblock_fidelity", "Reads as a flat-ink woodblock print, not a filtered photo."),
    ("layer_economy", "Achieves the image in very few blocks/colors."),
    ("masters_resemblance", "Evokes Hiroshige / Hokusai / Killion sensibility."),
]
