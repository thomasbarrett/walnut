from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

from walnut.layers.linear import FusedLinear
from walnut.models.loader import copy_weights
from walnut.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from walnut.sampler import SamplingParams


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


# One `gate_up_proj` standing in for the checkpoint's `gate_proj` + `up_proj`.
_GATE, _UP = torch.ones(2, 2), torch.full((3, 2), 2.0)


def _fused_module() -> nn.Module:
    parts = {"gate_proj": 2, "up_proj": 3}
    return nn.ModuleDict(
        {"mlp": nn.ModuleDict({"gate_up_proj": FusedLinear(2, parts)})}
    )


def _weight(module: nn.Module) -> torch.Tensor:
    return dict(module.named_parameters())["mlp.gate_up_proj.weight"]


def test_copy_weights_concatenates_a_fused_parameter():
    module = _fused_module()
    sources = [("mlp.gate_proj.weight", _GATE), ("mlp.up_proj.weight", _UP)]
    copy_weights(module, sources)
    assert torch.equal(_weight(module), torch.cat([_GATE, _UP]))


def test_copy_weights_concatenates_in_declared_order_not_arrival_order():
    """The bug this guards: a checkpoint listing ``up`` first would load it as
    ``gate``, which is silent — the concatenation still has the right shape."""
    module = _fused_module()
    sources = [("mlp.up_proj.weight", _UP), ("mlp.gate_proj.weight", _GATE)]
    copy_weights(module, sources)
    assert torch.equal(_weight(module), torch.cat([_GATE, _UP]))


def test_copy_weights_rejects_a_fused_parameter_missing_a_source():
    module = _fused_module()
    with pytest.raises(ValueError, match="unfilled"):
        copy_weights(module, [("mlp.gate_proj.weight", _GATE)])


def test_copy_weights_fills_a_fused_bias():
    parts = {"q_proj": 2, "k_proj": 1}
    module = nn.ModuleDict(
        {"attn": nn.ModuleDict({"qkv_proj": FusedLinear(2, parts, bias=True)})}
    )
    q, k = torch.ones(2, 2), torch.full((1, 2), 2.0)
    qb, kb = torch.ones(2), torch.full((1,), 2.0)
    copy_weights(
        module,
        [
            ("attn.q_proj.weight", q),
            ("attn.k_proj.weight", k),
            ("attn.q_proj.bias", qb),
            ("attn.k_proj.bias", kb),
        ],
    )
    params = dict(module.named_parameters())
    assert torch.equal(params["attn.qkv_proj.weight"], torch.cat([q, k]))
    assert torch.equal(params["attn.qkv_proj.bias"], torch.cat([qb, kb]))


def test_fused_linear_splits_in_the_order_it_loads():
    """The bug this guards: the split widths and the load order living in two
    places, so a reordered checkpoint loads silently into the wrong slices.

    Loading an identity-per-part weight makes each part recoverable, so the
    tensor `forward` returns as ``gate`` must be the one loaded as
    ``gate_proj``."""
    module = _fused_module()
    copy_weights(
        module,
        [("mlp.gate_proj.weight", _GATE), ("mlp.up_proj.weight", _UP)],
    )
    fused = cast(FusedLinear, module.get_submodule("mlp.gate_up_proj"))
    gate, up = fused(torch.ones(1, 2))
    assert torch.equal(gate, _GATE.sum(dim=1).unsqueeze(0))
    assert torch.equal(up, _UP.sum(dim=1).unsqueeze(0))


class _ScriptedModel:
    """Stand-in that drives `iter_generate`'s loop off a fixed token list.

    Exercises the decode loop without weights: `iter_generate` is called as an
    unbound function, so only the attributes it touches need to exist.
    """

    def __init__(self, tokens: list[int], eos_token_id: int | None = None) -> None:
        self.tokens = list(tokens)
        self.eos_token_id = eos_token_id
        self.forwards = 0
        self.lm_head = SimpleNamespace(weight=torch.zeros(1))
        self.model = SimpleNamespace(
            language_model=SimpleNamespace(make_cache=lambda **kwargs: None)
        )

    def __call__(self, input_ids, positions, cache):
        self.forwards += 1
        return torch.zeros(1, 1, 1)

    def sampler(self, logits, params, generator):
        # The loop samples one token past the last one it yields; 0 stands in.
        return torch.tensor([[self.tokens.pop(0) if self.tokens else 0]])


def _generate(
    model: _ScriptedModel,
    max_new_tokens: int,
    compile: bool = False,
    autotune: bool = True,
    **kwargs,
) -> list[int]:
    params = SamplingParams(max_new_tokens=max_new_tokens, **kwargs)
    ids = torch.zeros(1, 3, dtype=torch.long)
    stand_in = cast(Qwen3_5ForConditionalGeneration, model)
    # Compilation is off unless a test is about it: these exercise the decode
    # loop, and tracing the stand-in would test dynamo instead.
    return list(
        Qwen3_5ForConditionalGeneration.iter_generate(
            stand_in, ids, params, compile=compile, autotune=autotune
        )
    )


def test_iter_generate_yields_sampled_tokens_in_order():
    model = _ScriptedModel([7, 8, 9])
    assert _generate(model, max_new_tokens=3) == [7, 8, 9]


def test_iter_generate_stops_after_yielding_a_stop_token():
    model = _ScriptedModel([7, 5, 9])
    assert _generate(model, max_new_tokens=3, stop_token_ids=(5,)) == [7, 5]


def test_iter_generate_falls_back_to_eos_as_the_stop_token():
    model = _ScriptedModel([7, 2, 9], eos_token_id=2)
    assert _generate(model, max_new_tokens=3) == [7, 2]


def test_iter_generate_compiles_the_decode_step_but_not_the_prefill(monkeypatch):
    """Prefill's shapes follow the prompt; compiling it recompiles per length."""
    compiled: list[tuple[object, int]] = []

    def fake_compile(target, mode=None):
        # The forward count at compile time says which phases it can cover.
        compiled.append((target, target.forwards))
        return target

    monkeypatch.setattr(torch, "compile", fake_compile)

    model = _ScriptedModel([7, 8, 9])
    assert _generate(model, max_new_tokens=3, compile=True) == [7, 8, 9]
    assert compiled == [(model, 1)]  # compiled once, after the prefill forward
    assert model.forwards == 4  # one prefill, three decode steps

    assert _generate(_ScriptedModel([7, 8, 9]), max_new_tokens=3) == [7, 8, 9]
    assert len(compiled) == 1  # compile=False does not compile


def test_autotune_picks_the_compile_mode_and_never_inductor_cudagraphs(monkeypatch):
    """`DecodeGraph` captures the step, so Inductor must not also capture it:
    a mode without "-no-cudagraphs" would have it graph a region walnut graphs
    again."""
    modes: list[object] = []

    def fake_compile(target, mode=None):
        modes.append(mode)
        return target

    monkeypatch.setattr(torch, "compile", fake_compile)

    _generate(_ScriptedModel([7]), max_new_tokens=1, compile=True, autotune=True)
    _generate(_ScriptedModel([7]), max_new_tokens=1, compile=True, autotune=False)
    assert modes == ["max-autotune-no-cudagraphs", None]

    # Without a compile there is nothing to tune, so autotune must not compile.
    _generate(_ScriptedModel([7]), max_new_tokens=1, compile=False, autotune=True)
    assert len(modes) == 2


def test_iter_generate_runs_each_phase_in_a_named_frame():
    """Traces are segmented by these names, so they are an interface.

    Inlining either function back into the loop would break every per-phase
    query and fail nothing else.
    """
    from torch.profiler import ProfilerActivity, profile

    model = _ScriptedModel([7, 8, 9])
    with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
        assert _generate(model, max_new_tokens=3) == [7, 8, 9]

    frames = [event.name for event in prof.events()]
    assert sum(name.endswith(": _prefill") for name in frames) == 1
    assert sum(name.endswith(": _decode_step") for name in frames) == 3
