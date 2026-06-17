"""Adapter registry for the stylization slot."""
from __future__ import annotations

from .base import StylizeAdapter
from .fal_adapter import FalAdapter
from .replicate_adapter import ReplicateAdapter
from .stub import StubStylizeAdapter

_ADAPTERS: dict[str, type[StylizeAdapter]] = {
    "stub": StubStylizeAdapter,
    "replicate": ReplicateAdapter,
    "fal": FalAdapter,
}


def get_adapter(name: str) -> StylizeAdapter:
    if name not in _ADAPTERS:
        raise KeyError(
            f"Unknown stylize adapter {name!r}. Available: {sorted(_ADAPTERS)}"
        )
    return _ADAPTERS[name]()


def available_adapters() -> list[str]:
    return sorted(_ADAPTERS)
