"""Claude LLM-judge for the benchmark.

Scores a candidate result (its stylized image and/or color preview) against the
source photo on the carvability/aesthetic criteria in prompts.JUDGE_CRITERIA.
Returns a JudgeScore. Falls back to a null score (with a note) when no key is set.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .config import Config
from .prompts import JUDGE_CRITERIA, JUDGE_SYSTEM


class CriterionScore(BaseModel):
    name: str
    score: float = Field(ge=0, le=10)
    comment: str


class JudgeScore(BaseModel):
    criteria: list[CriterionScore]
    overall: float = Field(ge=0, le=10, description="Weighted holistic score.")
    verdict: str = Field(description="One-line summary of strengths/weaknesses.")

    def mean(self) -> float:
        return self.overall


def judge_result(
    source_photo: str | Path,
    candidate_image: str | Path,
    cfg: Config,
    *,
    label: str = "",
) -> JudgeScore:
    if not cfg.claude_available:
        return JudgeScore(
            criteria=[CriterionScore(name=n, score=0.0, comment="no judge (offline)")
                      for n, _ in JUDGE_CRITERIA],
            overall=0.0,
            verdict="Offline — set ANTHROPIC_API_KEY to enable Claude judging.",
        )

    from ._claude import client, image_block

    criteria_text = "\n".join(f"- {n}: {desc}" for n, desc in JUDGE_CRITERIA)
    resp = client().messages.parse(
        model=cfg.claude_model,
        max_tokens=2000,
        thinking={"type": "adaptive"},
        system=JUDGE_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"SOURCE PHOTOGRAPH{' ('+label+')' if label else ''}:"},
                image_block(source_photo),
                {"type": "text", "text": "CANDIDATE WOODBLOCK REDUCTION:"},
                image_block(candidate_image),
                {"type": "text",
                 "text": f"Score each criterion 0-10 and give an overall:\n{criteria_text}"},
            ],
        }],
        output_format=JudgeScore,
    )
    score = resp.parsed_output
    if score is None:
        raise RuntimeError("Judge returned no parseable score.")
    return score
