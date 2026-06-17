"""Claude vision analysis: photograph -> BlockPlan.

Uses structured outputs (`messages.parse`) so Claude's reduction populates the
pydantic BlockPlan directly. Falls back to a deterministic heuristic plan when
no API key is configured, so the pipeline always runs.
"""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .models import BlockPlan, Layer, ProductionMode
from .prompts import ANALYSIS_SYSTEM, analysis_user_prompt


def analyze_photo(
    photo_path: str | Path,
    cfg: Config,
    *,
    mode: ProductionMode = ProductionMode.SEPARATE,
    target_layers: int = 4,
) -> BlockPlan:
    if not cfg.claude_available:
        return _heuristic_plan(Path(photo_path), target_layers, mode)

    from ._claude import client, image_block

    resp = client().messages.parse(
        model=cfg.claude_model,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=ANALYSIS_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                image_block(photo_path),
                {"type": "text",
                 "text": analysis_user_prompt(mode=mode.value, target_layers=target_layers)},
            ],
        }],
        output_format=BlockPlan,
    )
    plan = resp.parsed_output
    if plan is None:
        raise RuntimeError("Claude returned no parseable plan; check stop_reason/refusal.")
    _validate(plan)
    return plan


def _validate(plan: BlockPlan) -> None:
    # Enforce the one-key-block invariant; if Claude over/under-specified, repair.
    keys = [l for l in plan.layers if l.is_key_block]
    if len(keys) == 0 and plan.layers:
        plan.layers[-1].is_key_block = True   # darkest becomes the key block
    elif len(keys) > 1:
        for extra in keys[1:]:
            extra.is_key_block = False
    # A layer can't be both key and background.
    for l in plan.layers:
        if l.is_key_block and l.is_background:
            l.is_background = False
    # Must keep at least one PRINTED color layer (not key, not paper).
    printed = [l for l in plan.layers if not l.is_key_block and not l.is_background]
    if not printed:
        for l in plan.layers:
            if not l.is_key_block:
                l.is_background = False
                break


# --- offline fallback ------------------------------------------------------

def _heuristic_plan(photo: Path, target_layers: int, mode: ProductionMode) -> BlockPlan:
    """A sparse default plan without Claude: bare paper + 2 inks + key block."""
    layers = [
        Layer(name="paper (negative space)", order=0, hex_color="#efe9da",
              is_background=True, notes="left unprinted — the breathing room"),
        Layer(name="light ink", order=1, hex_color="#9bb0a6", opacity=0.9,
              split_fountain=True, notes="lightest flat color, printed first"),
        Layer(name="mid ink", order=2, hex_color="#5e7d74", opacity=0.95,
              notes="second flat color"),
        Layer(name="key block", order=3, hex_color="#20302b", is_key_block=True,
              notes="minimal outlines, printed last"),
    ]
    return BlockPlan(
        title=photo.stem.replace("_", " ").title(),
        mode=mode,
        palette_rationale="Offline sparse heuristic: bare paper + two flat inks + a "
                          "minimal key block (set ANTHROPIC_API_KEY for a real "
                          "Hiroshige/Hokusai/Killion analysis).",
        composition_notes="Mostly negative space; a couple of flat shapes; outlines "
                          "only at region seams.",
        layers=layers,
    )
