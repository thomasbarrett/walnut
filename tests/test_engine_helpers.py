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


def test_resolve_device_auto_follows_cuda_availability():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device().type == expected
    assert resolve_device("auto").type == expected


def test_resolve_device_passes_through_explicit_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_device("cpu") == CPU
    assert resolve_device("cuda:1") == torch.device("cuda:1")


def test_resolve_device_rejects_cuda_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError):
        resolve_device("cuda")


def test_resolve_device_rejects_out_of_range_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError):
        resolve_device("cuda:9")


def test_resolve_dtype_auto_uses_checkpoint_dtype():
    config = PretrainedConfig(dtype="bfloat16")
    assert resolve_dtype("auto", config, CPU) == torch.bfloat16
    assert resolve_dtype(None, config, CUDA) == torch.bfloat16


def test_resolve_dtype_auto_downcasts_float32_off_cpu():
    config = PretrainedConfig(dtype="float32")
    assert resolve_dtype("auto", config, CUDA) == torch.bfloat16
    assert resolve_dtype("auto", config, CPU) == torch.float32


def test_resolve_dtype_auto_defaults_to_float32_without_declaration():
    assert resolve_dtype("auto", PretrainedConfig(), CPU) == torch.float32


def test_resolve_dtype_explicit_overrides_checkpoint():
    config = PretrainedConfig(dtype="bfloat16")
    assert resolve_dtype("float32", config, CPU) == torch.float32
    assert resolve_dtype(torch.float16, config, CPU) == torch.float16


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
