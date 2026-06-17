"""Adapter registry for the stylization slot."""
from __future__ import annotations

from .base import StylizeAdapter
from .stub import StubStylizeAdapter

_ADAPTERS: dict[str, type[StylizeAdapter]] = {
    "stub": StubStylizeAdapter,
    # "myprovider": MyProviderAdapter,   # <- register real adapters here
}


def get_adapter(name: str) -> StylizeAdapter:
    if name not in _ADAPTERS:
        raise KeyError(
            f"Unknown stylize adapter {name!r}. Available: {sorted(_ADAPTERS)}"
        )
    return _ADAPTERS[name]()


def available_adapters() -> list[str]:
    return sorted(_ADAPTERS)
