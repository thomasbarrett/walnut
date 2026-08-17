"""Shared checkpoint-to-module weight copying.

Architectures call `copy_weights` from their own ``load_weights`` hook,
supplying whatever name mapping or skip list their checkpoint needs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import nn


def _sample(names: list[str], limit: int = 3) -> str:
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown}, ... (+{len(names) - limit})"


def copy_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    skip_prefixes: Sequence[str] = (),
    fused: Mapping[str, Sequence[str]] = {},
) -> None:
    """Copy ``weights`` into ``module``'s parameters, matching by name.

    Strict in both directions, because a naming mismatch is otherwise silent:
    a parameter the checkpoint never fills keeps its random init, and a
    checkpoint tensor with nowhere to go is dropped. ``skip_prefixes`` names
    the checkpoint entries a model knowingly ignores, so they stay a
    deliberate choice rather than an accident.

    ``fused`` maps a parameter-name suffix to the checkpoint suffixes whose
    tensors concatenate along dim 0 to fill it, so a module that runs several
    of a checkpoint's projections as one `nn.Linear` still loads the checkpoint
    unmodified. Sources are held until every slot of a fused parameter has
    arrived — the checkpoint does not promise an order — and a fused parameter
    missing any source is reported like any other unfilled one.
    """
    params = dict(module.named_parameters())
    filled: set[str] = set()
    unmatched: list[str] = []
    pending: dict[str, list[torch.Tensor | None]] = {}

    def slot(name: str) -> tuple[str, int, int] | None:
        """Locate ``name`` as source ``i`` of ``n`` for some fused parameter."""
        for target, sources in fused.items():
            for i, source in enumerate(sources):
                if name.endswith("." + source):
                    param_name = name[: -len(source)] + target
                    if param_name in params:
                        return param_name, i, len(sources)
        return None

    for name, tensor in weights:
        param = params.get(name)
        if param is None:
            if (found := slot(name)) is not None:
                param_name, i, n = found
                pending.setdefault(param_name, [None] * n)[i] = tensor
            elif not name.startswith(tuple(skip_prefixes)):
                unmatched.append(name)
            continue
        param.data.copy_(tensor)
        filled.add(name)

    for param_name, parts in pending.items():
        present = [part for part in parts if part is not None]
        if len(present) < len(parts):
            continue
        params[param_name].data.copy_(torch.cat(present, dim=0))
        filled.add(param_name)

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
