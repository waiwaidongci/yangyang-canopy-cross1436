"""Compatibility alias: analysis runs are the persisted evaluation records."""

from .run import Run

Evaluation = Run

__all__ = ["Evaluation", "Run"]
