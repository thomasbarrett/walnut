import click
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


def test_serve_passes_cuda_graph_through(monkeypatch):
    seen: dict[str, bool] = {}

    class _Engine:
        model_id = "m"
        device = "cpu"
        dtype = "float32"

    def fake_load_model(model, device=None, dtype=None, cuda_graph=True):
        seen["cuda_graph"] = cuda_graph
        return _Engine()

    monkeypatch.setattr("walnut.engine.load_model", fake_load_model)
    monkeypatch.setattr("walnut.server.serve", lambda *args, **kwargs: None)

    assert runner.invoke(app, ["serve", "m"]).exit_code == 0
    assert seen["cuda_graph"] is True

    assert runner.invoke(app, ["serve", "m", "--no-cuda-graph"]).exit_code == 0
    assert seen["cuda_graph"] is False


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
