"""Per-layer decode cache marker.

Each token mixer creates and consumes its own subclass (`KVCache` for full
attention, `ConvState` for linear attention); `Cache` lets the model hold a
``list[Cache]`` without naming both.
"""

from __future__ import annotations


class Cache:
    """Opaque per-layer decode state; only the owning mixer knows its shape."""
