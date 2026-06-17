"""Benchmark harness: sweep approaches/prompts/params over a photo set."""
from __future__ import annotations

from .matrix import Cell, default_matrix
from .run import run_benchmark

__all__ = ["Cell", "default_matrix", "run_benchmark"]
