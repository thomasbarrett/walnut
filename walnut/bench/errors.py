"""The one exception the benchmark commands raise for a bad run."""

from __future__ import annotations


class BenchError(RuntimeError):
    """A run whose numbers must not be quoted, with the reason why."""
