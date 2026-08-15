import gzip
import json
import threading

import pytest
import torch
from fastapi.testclient import TestClient

from tests.conftest import StubEngine
from walnut.profiler import DIR_ENV, TorchProfiler
from walnut.server import create_app


def _work() -> None:
    """Tensor work, so the window records something."""
    a = torch.randn(64, 64)
    (a @ a).sum().item()


def test_from_env_without_the_variable():
    assert TorchProfiler.from_env(env={}) is None


def test_from_env_reads_the_directory(tmp_path):
    profiler = TorchProfiler.from_env(env={DIR_ENV: str(tmp_path)})
    assert profiler is not None
    assert profiler.directory == tmp_path


def test_from_env_leaves_stacks_off(tmp_path):
    """Stacks break summarize on the server's traces."""
    profiler = TorchProfiler.from_env(env={DIR_ENV: str(tmp_path)})
    assert profiler is not None
    assert profiler.with_stack is False


def test_stop_writes_trace_and_summary(tmp_path):
    profiler = TorchProfiler(tmp_path, with_stack=False)
    profiler.start()
    _work()
    artifacts = profiler.stop()

    assert artifacts.trace.parent == tmp_path
    assert artifacts.trace.name.endswith(".trace.json.gz")
    with gzip.open(artifacts.trace, "rt") as handle:
        trace = json.load(handle)
    # Chrome traces are an object with traceEvents, or a bare event list.
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    assert events

    assert artifacts.summary is not None
    summary = artifacts.summary.read_text()
    assert "Self CPU" in summary
    assert "Name" in summary


def test_unbuildable_summary_still_yields_a_trace(tmp_path, monkeypatch):
    """Parsing kineto results can fail on a profile that exported fine."""

    def boom(profile, row_limit=40):
        raise UnicodeDecodeError("utf-8", b"\x80", 0, 1, "invalid start byte")

    monkeypatch.setattr("walnut.profiler.summarize", boom)
    profiler = TorchProfiler(tmp_path, with_stack=False)
    profiler.start()
    _work()
    artifacts = profiler.stop()

    assert artifacts.summary is None
    assert artifacts.trace.exists()
    assert artifacts.as_dict()["summary"] is None


def test_start_is_not_reentrant(tmp_path):
    profiler = TorchProfiler(tmp_path, with_stack=False)
    profiler.start()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            profiler.start()
    finally:
        profiler.stop()


def test_stop_without_start_raises(tmp_path):
    with pytest.raises(RuntimeError, match="not running"):
        TorchProfiler(tmp_path).stop()


def test_running_tracks_the_window(tmp_path):
    profiler = TorchProfiler(tmp_path, with_stack=False)
    assert not profiler.running
    profiler.start()
    assert profiler.running
    profiler.stop()
    assert not profiler.running


def test_directory_is_created_on_stop(tmp_path):
    target = tmp_path / "nested" / "profiles"
    profiler = TorchProfiler(target, with_stack=False)
    profiler.start()
    _work()
    assert profiler.stop().trace.exists()


def test_stopping_from_another_thread_raises(tmp_path):
    """Crossing threads segfaults kineto, so it is refused up front."""
    profiler = TorchProfiler(tmp_path, with_stack=False)
    profiler.start()
    error: list[BaseException] = []

    def stop_elsewhere() -> None:
        try:
            profiler.stop()
        except RuntimeError as exc:
            error.append(exc)

    thread = threading.Thread(target=stop_elsewhere)
    thread.start()
    thread.join()

    assert error and "thread that started it" in str(error[0])
    # The refusal left the window open, still owned by this thread.
    assert profiler.running
    profiler.stop()


def test_endpoints_404_when_disabled():
    client = TestClient(create_app(StubEngine()))
    assert client.post("/start_profile").status_code == 404
    assert client.post("/stop_profile").status_code == 404


def test_stop_without_start_is_a_conflict(tmp_path):
    profiler = TorchProfiler(tmp_path, with_stack=False)
    client = TestClient(create_app(StubEngine(), profiler=profiler))
    assert client.post("/stop_profile").status_code == 409
