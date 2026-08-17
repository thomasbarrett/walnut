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

# Every command that loads a model takes these, spelled the same way: a flag
# that means one thing under `serve` and another under `profile` is a trap.
Model = Annotated[
    str,
    typer.Argument(
        metavar="MODEL",
        help="Hugging Face model id or local path.",
    ),
]
Device = Annotated[
    str,
    typer.Option(
        envvar="WALNUT_DEVICE",
        help="Device to run on: 'auto' (CUDA when available), 'cpu', "
        "'cuda', 'cuda:1', ...",
    ),
]
Dtype = Annotated[
    str,
    typer.Option(
        envvar="WALNUT_DTYPE",
        help="Weight/activation dtype: 'auto' (the checkpoint's own dtype, "
        "downcast from float32 on accelerators), 'bfloat16', 'float16', "
        "'float32'.",
    ),
]
CudaGraph = Annotated[
    bool,
    typer.Option(
        "--cuda-graph/--no-cuda-graph",
        envvar="WALNUT_CUDA_GRAPH",
        help="Replay decode from a captured CUDA graph, dropping the "
        "per-token launch cost. Ignored off CUDA.",
    ),
]
Compile = Annotated[
    bool,
    typer.Option(
        "--compile/--no-compile",
        envvar="WALNUT_COMPILE",
        help="Run the decode step through torch.compile, fusing its "
        "elementwise kernels. Costs a few seconds on the first request.",
    ),
]
Autotune = Annotated[
    bool,
    typer.Option(
        "--autotune/--no-autotune",
        envvar="WALNUT_AUTOTUNE",
        help="Have the compile benchmark a Triton kernel against cuBLAS for "
        "each projection, rather than take cuBLAS on faith. Worth ~11% of "
        "TPOT at decode's batch of 1. Costs a longer first compile, cached on "
        "disk thereafter. Ignored with --no-compile.",
    ),
]


@app.command()
def serve(
    model: Model,
    host: Annotated[
        str,
        typer.Option(
            envvar="WALNUT_HOST", help="Host/interface to bind the server to."
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int, typer.Option(envvar="WALNUT_PORT", help="Port to listen on.")
    ] = 8000,
    device: Device = "auto",
    dtype: Dtype = "auto",
    cuda_graph: CudaGraph = True,
    compile: Compile = True,
    autotune: Autotune = True,
) -> None:
    """Serve MODEL behind an OpenAI-compatible API."""
    from .engine import load_model, parse_dtype, resolve_device
    from .server import serve as run_server

    # Resolve the flags before the (slow) load, so a typo fails fast and a real
    # load failure surfaces as itself rather than as a bad-parameter error.
    try:
        target = resolve_device(device)
        precision = parse_dtype(dtype)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Loading '{model}'...")
    engine = load_model(
        model,
        device=target,
        dtype=precision,
        cuda_graph=cuda_graph,
        compile=compile,
        autotune=autotune,
    )
    typer.echo(
        f"Serving '{engine.model_id}' on http://{host}:{port}/v1 "
        f"({engine.device}, {str(engine.dtype).removeprefix('torch.')})"
    )
    run_server(engine, host=host, port=port)


@app.command()
def profile(
    model: Model,
    prompt: Annotated[
        str, typer.Option(help="Prompt to generate from while profiling.")
    ] = "Explain how a transformer works.",
    max_tokens: Annotated[
        int, typer.Option(help="Tokens to decode in the profiled window.")
    ] = 32,
    output_dir: Annotated[
        str,
        typer.Option(
            envvar="WALNUT_TORCH_PROFILER_DIR",
            help="Directory to write the trace and summary to.",
        ),
    ] = "./profiles",
    temperature: Annotated[
        float,
        typer.Option(
            help="Sampling temperature for the profiled generation. Defaults to "
            "greedy, matching the benchmark; raise it to see the sampler's own "
            "kernels, which greedy decoding never runs.",
        ),
    ] = 0.0,
    device: Device = "auto",
    dtype: Dtype = "auto",
    cuda_graph: CudaGraph = True,
    compile: Compile = True,
    autotune: Autotune = True,
) -> None:
    """Profile one generation with MODEL and write a Chrome trace.

    The trace opens at https://ui.perfetto.dev/; the summary beside it is the
    same run as text.
    """
    from .engine import GenerationConfig, Message, load_model, parse_dtype
    from .engine import resolve_device as resolve
    from .profiler import TorchProfiler

    try:
        target = resolve(device)
        precision = parse_dtype(dtype)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Loading '{model}'...")
    engine = load_model(
        model,
        device=target,
        dtype=precision,
        cuda_graph=cuda_graph,
        compile=compile,
        autotune=autotune,
    )
    config = GenerationConfig(max_tokens=max_tokens, temperature=temperature)
    messages = [Message(role="user", content=prompt)]

    # A cold pass pays for autotuning and lazy init, swamping the real numbers.
    typer.echo("Warming up...")
    engine.generate(messages, GenerationConfig(max_tokens=4, temperature=temperature))

    typer.echo(f"Profiling {max_tokens} tokens on {engine.device}...")
    profiler = TorchProfiler(output_dir)
    profiler.start()
    try:
        engine.generate(messages, config)
    finally:
        artifacts = profiler.stop()

    typer.echo(f"trace:   {artifacts.trace}  (open at https://ui.perfetto.dev/)")
    if artifacts.summary is not None:
        typer.echo(f"summary: {artifacts.summary}")
    else:
        typer.echo("summary: unavailable (see the log)")


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
