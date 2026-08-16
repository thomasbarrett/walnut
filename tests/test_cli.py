import click
import typer.main
from typer.testing import CliRunner

from walnut.cli import app

runner = CliRunner()

# Typer renders help through Rich, which styles option names with ANSI color.
# On CI (color forced) the two leading dashes land in separate escape spans, so
# "--host" isn't a literal substring — unstyle before matching.


def test_serve_help_lists_arguments():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    out = click.unstyle(result.output)
    assert "MODEL" in out
    for opt in ("--host", "--port", "--device", "--dtype", "--no-cuda-graph"):
        assert opt in out


def test_serve_passes_the_speed_flags_through(monkeypatch):
    seen: dict[str, bool] = {}

    class _Engine:
        model_id = "m"
        device = "cpu"
        dtype = "float32"

    def fake_load_model(model, device=None, dtype=None, cuda_graph=True, compile=True):
        seen.update(cuda_graph=cuda_graph, compile=compile)
        return _Engine()

    monkeypatch.setattr("walnut.engine.load_model", fake_load_model)
    monkeypatch.setattr("walnut.server.serve", lambda *args, **kwargs: None)

    assert runner.invoke(app, ["serve", "m"]).exit_code == 0
    assert seen == {"cuda_graph": True, "compile": True}

    invoked = runner.invoke(app, ["serve", "m", "--no-cuda-graph", "--no-compile"])
    assert invoked.exit_code == 0
    assert seen == {"cuda_graph": False, "compile": False}


def _params(command: str) -> dict[str, click.Parameter]:
    group = typer.main.get_command(app)
    subcommand = group.get_command(click.Context(group), command)  # ty: ignore
    assert subcommand is not None
    return {p.name: p for p in subcommand.params if p.name}


def test_model_loading_options_match_between_serve_and_profile():
    """`serve` and `profile` load the same engine, so they must ask alike."""
    serve_params = _params("serve")
    profile_params = _params("profile")

    for name in ("model", "device", "dtype", "cuda_graph", "compile"):
        mine, theirs = serve_params[name], profile_params[name]
        assert mine.opts == theirs.opts, name
        assert mine.default == theirs.default, name
        assert mine.envvar == theirs.envvar, name
        assert getattr(mine, "help", None) == getattr(theirs, "help", None), name


def test_profile_defaults_to_greedy_so_it_matches_the_benchmark():
    """The benchmark samples greedily; a profile at another temperature would
    trace the sampler's softmax, which the benchmarked path never runs."""
    temperature = _params("profile")["temperature"]
    assert temperature.default == 0.0


def test_profile_passes_the_speed_flags_through(monkeypatch):
    seen: dict[str, bool] = {}

    def fake_load_model(model, device=None, dtype=None, cuda_graph=True, compile=True):
        seen.update(cuda_graph=cuda_graph, compile=compile)
        raise RuntimeError("stop before the profiled run")

    monkeypatch.setattr("walnut.engine.load_model", fake_load_model)

    runner.invoke(app, ["profile", "m"])
    assert seen == {"cuda_graph": True, "compile": True}

    runner.invoke(app, ["profile", "m", "--no-cuda-graph", "--no-compile"])
    assert seen == {"cuda_graph": False, "compile": False}


def test_chat_help_lists_options():
    result = runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0
    out = click.unstyle(result.output)
    for opt in ("--model", "--url", "--quick"):
        assert opt in out


def test_chat_quick_prints_completion_and_exits(live_server):
    result = runner.invoke(app, ["chat", "--url", live_server, "--quick", "ping"])
    assert result.exit_code == 0
    assert "ping" in result.output


def test_chat_defaults_to_first_model(live_server):
    # No --model: should pick the backend's first model without error.
    result = runner.invoke(app, ["chat", "--url", live_server, "--quick", "hello"])
    assert result.exit_code == 0
    assert "hello" in result.output


def test_chat_reports_unreachable_backend():
    result = runner.invoke(
        app, ["chat", "--url", "http://127.0.0.1:1/v1", "--quick", "hi"]
    )
    assert result.exit_code != 0
