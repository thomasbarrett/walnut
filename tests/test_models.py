import pytest
import torch
from torch import nn

from walnut.models.loader import copy_weights


def _module() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))


def _weights(module: nn.Module) -> list[tuple[str, torch.Tensor]]:
    return [(name, torch.ones_like(p)) for name, p in module.named_parameters()]


def test_copy_weights_fills_every_parameter():
    module = _module()
    copy_weights(module, _weights(module))
    assert all((p == 1).all() for p in module.parameters())


def test_copy_weights_rejects_an_unfilled_parameter():
    # The bug this guards: a renamed parameter silently keeps its random init.
    module = _module()
    with pytest.raises(ValueError, match="unfilled"):
        copy_weights(module, _weights(module)[:1])


def test_copy_weights_rejects_an_unmatched_tensor():
    module = _module()
    extra = [*_weights(module), ("2.weight", torch.ones(2, 2))]
    with pytest.raises(ValueError, match="unmatched"):
        copy_weights(module, extra)


def test_copy_weights_skips_listed_prefixes():
    module = _module()
    extra = [*_weights(module), ("mtp.fc.weight", torch.ones(2, 2))]
    copy_weights(module, extra, skip_prefixes=("mtp.",))
    assert all((p == 1).all() for p in module.parameters())
