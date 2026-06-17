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
    target_layers: int = 5,
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
        # Darkest layer (highest order) becomes the key block.
        plan.layers[-1].is_key_block = True
    elif len(keys) > 1:
        for extra in keys[1:]:
            extra.is_key_block = False


# --- offline fallback ------------------------------------------------------

def _heuristic_plan(photo: Path, target_layers: int, mode: ProductionMode) -> BlockPlan:
    """A reasonable default plan without Claude: a Killion-ish alpine palette."""
    palette = ["#e7e2d3", "#a9c0cf", "#6f97a8", "#3f6b6e", "#23373a"]
    n = max(2, min(target_layers, len(palette)))
    layers = [
        Layer(name=f"value {i}", order=i, hex_color=palette[i],
              opacity=0.9 if i < n - 1 else 1.0,
              split_fountain=(i == 1),  # treat the high sky band as split-fountain
              is_key_block=(i == n - 1),
              notes="heuristic layer (no Claude analysis available)")
        for i in range(n)
    ]
    return BlockPlan(
        title=photo.stem.replace("_", " ").title(),
        mode=mode,
        palette_rationale="Offline heuristic alpine palette (set ANTHROPIC_API_KEY "
                          "for a real Killion-style analysis).",
        composition_notes="Posterized into value bands; darkest band carries the key block.",
        layers=layers,
    )
