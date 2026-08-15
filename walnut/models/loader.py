"""Shared checkpoint-to-module weight copying.

Architectures call `copy_weights` from their own ``load_weights`` hook,
supplying whatever name mapping or skip list their checkpoint needs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn


def _sample(names: list[str], limit: int = 3) -> str:
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown}, ... (+{len(names) - limit})"


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
    """
    params = dict(module.named_parameters())
    filled: set[str] = set()
    unmatched: list[str] = []
    for name, tensor in weights:
        param = params.get(name)
        if param is None:
            if not name.startswith(tuple(skip_prefixes)):
                unmatched.append(name)
            continue
        param.data.copy_(tensor)
        filled.add(name)

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
