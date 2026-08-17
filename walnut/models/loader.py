"""Shared checkpoint-to-module weight copying.

Architectures call `copy_weights` from their own ``load_weights`` hook,
supplying whatever name mapping or skip list their checkpoint needs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn

from walnut.layers.linear import FusedLinear


def _sample(names: list[str], limit: int = 3) -> str:
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown}, ... (+{len(names) - limit})"


def _fused_slots(module: nn.Module) -> dict[str, tuple[str, int, int]]:
    """Map each checkpoint name a `FusedLinear` absorbs to the slot it fills.

    A fused projection stands where the checkpoint keeps siblings, so
    ``model.layers.0.mlp.gate_up_proj.weight`` is filled by
    ``model.layers.0.mlp.{gate,up}_proj.weight``. Reading this off the modules
    keeps the fusion declared once, in the layer that splits it, rather than in
    a table beside the model that nothing checks against the split.
    """
    slots: dict[str, tuple[str, int, int]] = {}
    for path, child in module.named_modules():
        if not isinstance(child, FusedLinear):
            continue
        prefix = path.rsplit(".", 1)[0] + "." if "." in path else ""
        for param, _ in child.named_parameters(recurse=False):
            target = f"{path}.{param}" if path else param
            for i, part in enumerate(child.parts):
                slots[f"{prefix}{part}.{param}"] = (target, i, len(child.parts))
    return slots


def copy_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    skip_prefixes: Sequence[str] = (),
) -> None:
    """Copy ``weights`` into ``module``'s parameters, matching by name.

    Strict in both directions, because a naming mismatch is otherwise silent:
    a parameter the checkpoint never fills keeps its random init, and a
    checkpoint tensor with nowhere to go is dropped. ``skip_prefixes`` names
    the checkpoint entries a model knowingly ignores, so they stay a
    deliberate choice rather than an accident.

    A `FusedLinear` holds several of the checkpoint's projections as one
    parameter; its parts are collected and concatenated along dim 0, so a
    checkpoint loads unmodified into a model that fuses. Parts are held until
    the group is complete — the checkpoint does not promise an order — and a
    fused parameter missing any part is reported like any other unfilled one.
    """
    params = dict(module.named_parameters())
    slots = _fused_slots(module)
    skip = tuple(skip_prefixes)
    filled: set[str] = set()
    unmatched: list[str] = []
    pending: dict[str, list[torch.Tensor | None]] = {}

    for name, tensor in weights:
        if (param := params.get(name)) is not None:
            param.data.copy_(tensor)
            filled.add(name)
        elif (slot := slots.get(name)) is not None:
            target, i, count = slot
            pending.setdefault(target, [None] * count)[i] = tensor
        elif not name.startswith(skip):
            unmatched.append(name)

    for target, parts in pending.items():
        present = [part for part in parts if part is not None]
        if len(present) < len(parts):
            continue
        params[target].data.copy_(torch.cat(present, dim=0))
        filled.add(target)

    problems = []
    if missing := sorted(params.keys() - filled):
        problems.append(f"{len(missing)} unfilled parameter(s): {_sample(missing)}")
    if unmatched:
        problems.append(
            f"{len(unmatched)} unmatched checkpoint tensor(s): "
            f"{_sample(sorted(unmatched))}"
        )
    if problems:
        raise ValueError("checkpoint does not match the model — " + "; ".join(problems))
