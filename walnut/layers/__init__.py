"""Reusable model building blocks.

Only the primitives PyTorch doesn't provide live here; models use plain
``nn.Linear`` / ``nn.Embedding`` for everything else.
"""

from walnut.layers.attention import Attention
from walnut.layers.linear import FusedLinear
from walnut.layers.linear_attention import GatedDeltaNet
from walnut.layers.norm import RMSNorm
from walnut.layers.rotary import RotaryEmbedding, apply_rotary_pos_emb
from walnut.layers.vision import (
    VisionAttention,
    VisionPatchEmbed,
    VisionPatchMerger,
    VisionRotaryEmbedding,
)

__all__ = [
    "Attention",
    "FusedLinear",
    "GatedDeltaNet",
    "RMSNorm",
    "RotaryEmbedding",
    "VisionAttention",
    "VisionPatchEmbed",
    "VisionPatchMerger",
    "VisionRotaryEmbedding",
    "apply_rotary_pos_emb",
]
