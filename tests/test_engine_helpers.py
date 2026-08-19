import pytest
import torch
from transformers import PretrainedConfig

from walnut.engine import (
    _build_on,
    _first_stop,
    parse_dtype,
    resolve_device,
    resolve_dtype,
)
from walnut.models import resolve_model_class
from walnut.models.qwen3_5 import Qwen3_5ForConditionalGeneration

CPU = torch.device("cpu")
CUDA = torch.device("cuda")


def test_first_stop_returns_earliest_match():
    # "world" (index 6) precedes "stop" (index 12); the earliest wins.
    assert _first_stop("hello world stop here", ["stop", "world"]) == 6


def test_first_stop_none_when_no_match():
    assert _first_stop("hello", ["xyz"]) is None


def test_first_stop_none_for_empty_or_missing_stops():
    assert _first_stop("hello", None) is None
    assert _first_stop("hello", []) is None


def test_first_stop_ignores_empty_stop_strings():
    assert _first_stop("hello", [""]) is None


def _pretend_cuda(monkeypatch, count=2, capability=(9, 0)):
    """A machine with CUDA, for the checks that are about the selection."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _=None: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _=None: "Pretend GPU")


def test_resolve_device_auto_selects_cuda(monkeypatch):
    _pretend_cuda(monkeypatch)
    assert resolve_device() == CUDA
    assert resolve_device("auto") == CUDA


def test_resolve_device_passes_through_an_explicit_index(monkeypatch):
    _pretend_cuda(monkeypatch)
    assert resolve_device("cuda:1") == torch.device("cuda:1")


def test_resolve_device_rejects_cpu():
    """walnut decodes through a CUDA-only attention kernel, so a CPU selection
    has to fail here rather than inside the first request."""
    with pytest.raises(ValueError, match="CUDA"):
        resolve_device("cpu")


def test_resolve_device_rejects_a_gpu_older_than_the_kernel(monkeypatch):
    _pretend_cuda(monkeypatch, capability=(7, 5))
    with pytest.raises(ValueError, match="Ampere"):
        resolve_device("cuda")


def test_resolve_device_rejects_cuda_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError):
        resolve_device("cuda")


def test_resolve_device_rejects_out_of_range_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError):
        resolve_device("cuda:9")


def _bf16(monkeypatch, supported=True):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: supported)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _=None: "Pretend GPU")


def test_resolve_dtype_auto_uses_checkpoint_dtype(monkeypatch):
    _bf16(monkeypatch)
    config = PretrainedConfig(dtype="bfloat16")
    assert resolve_dtype("auto", config, CUDA) == torch.bfloat16
    assert resolve_dtype(None, config, CUDA) == torch.bfloat16


def test_resolve_dtype_auto_downcasts_a_float32_checkpoint(monkeypatch):
    """The attention kernel has no float32 build, so auto cannot leave one
    alone the way it could when there was a fallback."""
    _bf16(monkeypatch)
    assert resolve_dtype("auto", PretrainedConfig(dtype="float32"), CUDA) == (
        torch.bfloat16
    )
    assert resolve_dtype("auto", PretrainedConfig(), CUDA) == torch.bfloat16


def test_resolve_dtype_rejects_float32_asked_for_by_name(monkeypatch):
    """Downcasting silently is right for a checkpoint's own declaration and
    wrong for a flag: the caller asked for a precision walnut cannot serve."""
    _bf16(monkeypatch)
    with pytest.raises(ValueError, match="float32"):
        resolve_dtype("float32", PretrainedConfig(dtype="bfloat16"), CUDA)


def test_resolve_dtype_explicit_overrides_checkpoint(monkeypatch):
    _bf16(monkeypatch)
    config = PretrainedConfig(dtype="bfloat16")
    assert resolve_dtype(torch.float16, config, CUDA) == torch.float16


def test_resolve_dtype_falls_back_to_float16_without_bf16(monkeypatch):
    _bf16(monkeypatch, supported=False)
    config = PretrainedConfig(dtype="bfloat16")
    with pytest.warns(UserWarning, match="bfloat16 is unsupported"):
        assert resolve_dtype("auto", config, CUDA) == torch.float16


def test_parse_dtype_defers_auto_to_the_checkpoint():
    assert parse_dtype(None) is None
    assert parse_dtype("auto") is None


def test_parse_dtype_accepts_names_and_dtypes():
    assert parse_dtype("bfloat16") == torch.bfloat16
    assert parse_dtype(torch.float16) == torch.float16


def test_parse_dtype_rejects_non_float_selections():
    # No config needed, so the CLI can reject a typo before downloading weights.
    for bad in ("int8", "not_a_dtype", torch.int64):
        with pytest.raises(ValueError):
            parse_dtype(bad)


def test_build_on_applies_and_restores_defaults():
    before = torch.get_default_dtype()
    with _build_on(CPU, torch.bfloat16):
        assert torch.nn.Linear(2, 2).weight.dtype == torch.bfloat16
    assert torch.get_default_dtype() == before


def test_build_on_restores_defaults_after_a_failed_build():
    before = torch.get_default_dtype()
    with pytest.raises(RuntimeError), _build_on(CPU, torch.bfloat16):
        raise RuntimeError("construction blew up")
    assert torch.get_default_dtype() == before


def test_resolve_model_class_maps_registered_architecture():
    config = PretrainedConfig(architectures=["Qwen3_5ForConditionalGeneration"])
    assert resolve_model_class(config) is Qwen3_5ForConditionalGeneration


def test_resolve_model_class_unknown_architecture_raises():
    with pytest.raises(ValueError):
        resolve_model_class(PretrainedConfig(architectures=["NotARealArchitecture"]))


def test_resolve_model_class_missing_architectures_raises():
    with pytest.raises(ValueError):
        resolve_model_class(PretrainedConfig())


def test_parse_dtype_rejects_a_precision_the_kernel_lacks():
    """The bug this guards: rejecting float32 only in `resolve_dtype` puts the
    error after `AutoConfig.from_pretrained`, so a mistyped flag costs a
    download and arrives as a traceback instead of a bad-parameter message."""
    with pytest.raises(ValueError, match="unsupported"):
        parse_dtype("float32")
    with pytest.raises(ValueError, match="unsupported"):
        parse_dtype(torch.float64)
