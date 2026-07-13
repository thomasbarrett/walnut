"""The ``walnut`` command-line interface."""

from __future__ import annotations

from typing import Annotated

import typer

app = typer.Typer(
    help="walnut — an inference engine, built on PyTorch.",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_URL = "http://127.0.0.1:8000/v1"


@app.command()
def serve(
    model: Annotated[
        str,
        typer.Argument(
            metavar="MODEL",
            help="Hugging Face model id or local path to serve.",
        ),
    ],
    host: Annotated[
        str,
        typer.Option(
            envvar="WALNUT_HOST", help="Host/interface to bind the server to."
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int, typer.Option(envvar="WALNUT_PORT", help="Port to listen on.")
    ] = 8000,
) -> None:
    """Serve MODEL behind an OpenAI-compatible API."""
    from .engine import load_model
    from .server import serve as run_server

    typer.echo(f"Loading '{model}'...")
    engine = load_model(model)
    typer.echo(f"Serving '{engine.model_id}' on http://{host}:{port}/v1")
    run_server(engine, host=host, port=port)


@app.command()
def chat(
    model: Annotated[
        str | None,
        typer.Option(
            help="Model to chat with. Defaults to the first model the backend reports."
        ),
    ] = None,
    url: Annotated[
        str, typer.Option(help="Base URL of the OpenAI-compatible backend.")
    ] = DEFAULT_URL,
    quick: Annotated[
        str | None,
        typer.Option(
            help="Send a single prompt, print the completion, and exit.",
        ),
    ] = None,
) -> None:
    """Chat with a served model over an OpenAI-compatible API."""
    import httpx

    from .client import ChatClient

    with ChatClient(url) as client:
        try:
            selected = model or client.default_model()
        except (httpx.HTTPError, RuntimeError) as exc:
            raise typer.BadParameter(
                f"Could not reach backend at {url}: {exc}"
            ) from exc

        if quick is not None:
            reply = client.chat(selected, [{"role": "user", "content": quick}])
            typer.echo(reply)
            raise typer.Exit()

        _repl(client, selected)


def _repl(client, model: str) -> None:
    """Run an interactive chat loop until EOF or an empty exit command."""
    typer.echo(f"Chatting with '{model}'. Press Ctrl-D or type /exit to quit.")
    history: list[dict[str, str]] = []
    while True:
        try:
            user = typer.prompt("you", prompt_suffix=" > ")
        except (EOFError, typer.Abort):
            typer.echo()
            break
        if user.strip() in {"/exit", "/quit"}:
            break
        history.append({"role": "user", "content": user})
        typer.echo("assistant > ", nl=False)
        pieces: list[str] = []
        for piece in client.stream_chat(model, history):
            pieces.append(piece)
            typer.echo(piece, nl=False)
        typer.echo()
        history.append({"role": "assistant", "content": "".join(pieces)})
