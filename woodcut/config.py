"""Runtime configuration, loaded from environment (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# The most capable Opus-tier model — drives vision analysis and the LLM-judge.
DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str | None
    claude_model: str
    stylize_adapter: str

    @property
    def claude_available(self) -> bool:
        return bool(self.anthropic_api_key)


def load_config() -> Config:
    return Config(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        claude_model=os.environ.get("WOODCUT_CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
        stylize_adapter=os.environ.get("WOODCUT_STYLIZE_ADAPTER", "stub"),
    )
