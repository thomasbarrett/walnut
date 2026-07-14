import pytest
from transformers import PretrainedConfig

from walnut.engine import _first_stop
from walnut.models import resolve_model_class
from walnut.models.qwen3_5 import Qwen3_5ForConditionalGeneration


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


def test_resolve_model_class_maps_registered_architecture():
    config = PretrainedConfig(architectures=["Qwen3_5ForConditionalGeneration"])
    assert resolve_model_class(config) is Qwen3_5ForConditionalGeneration


def test_resolve_model_class_unknown_architecture_raises():
    with pytest.raises(ValueError):
        resolve_model_class(PretrainedConfig(architectures=["NotARealArchitecture"]))


def test_resolve_model_class_missing_architectures_raises():
    with pytest.raises(ValueError):
        resolve_model_class(PretrainedConfig())
