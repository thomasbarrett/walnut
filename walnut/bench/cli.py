"""`walnut bench` — five subcommands, and the flags they share.

    serve       what clients see, under load, over HTTP
    throughput  what the engine does with everything at once, no HTTP
    latency     what the decode path costs, one stream, no scheduler
    startup     what it costs to get ready to serve, and nothing else
    sweep       `serve` up a ladder of rates, to find where capacity runs out

Picking the wrong one wastes the measurement. `serve` carries queueing,
batching, detokenization and HTTP — what a user experiences, and enough to
drown a 3% kernel win. `latency` excludes all of it.
"""

from __future__ import annotations

import asyncio
import math
from typing import Annotated

import typer

from walnut.bench.errors import BenchError
from walnut.bench.metrics import SLO_METRICS
from walnut.bench.offline import EngineOptions, run_latency, run_startup, run_throughput
from walnut.bench.online import ServeOptions, run_serve, run_sweep
from walnut.bench.workload import (
    DEFAULT_SHAPE,
    SHAPES,
    goodput_config,
    resolve_shape,
)

app = typer.Typer(
    help="Measure walnut: latency, throughput, capacity and start-up.",
    no_args_is_help=True,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


def _sizes(shape) -> str:
    if shape.input_len is None:
        return f"fixed prompt, {shape.output_len} out"
    return f"{shape.input_len} in / {shape.output_len} out"


# -- shared option types ----------------------------------------------------

Model = Annotated[
    str, typer.Argument(metavar="MODEL", help="Hugging Face model id or local path.")
]
Device = Annotated[
    str,
    typer.Option(
        envvar="WALNUT_DEVICE",
        help="Device to run on: 'auto' (CUDA when available), 'cpu', 'cuda', ...",
    ),
]
Dtype = Annotated[
    str,
    typer.Option(
        envvar="WALNUT_DTYPE",
        help="Weight/activation dtype: 'auto', 'bfloat16', 'float16', 'float32'.",
    ),
]
CudaGraph = Annotated[
    bool,
    typer.Option(
        "--cuda-graph/--no-cuda-graph",
        envvar="WALNUT_CUDA_GRAPH",
        help="Replay decode from a captured CUDA graph. Ignored off CUDA.",
    ),
]
Compile = Annotated[
    bool,
    typer.Option(
        "--compile/--no-compile",
        envvar="WALNUT_COMPILE",
        help="Run the decode step through torch.compile.",
    ),
]
Autotune = Annotated[
    bool,
    typer.Option(
        "--autotune/--no-autotune",
        envvar="WALNUT_AUTOTUNE",
        help="Benchmark a Triton kernel against cuBLAS per projection. "
        "Ignored with --no-compile.",
    ),
]
NumIters = Annotated[int, typer.Option(min=1, help="Measured iterations.")]
NumItersWarmup = Annotated[
    int,
    typer.Option(
        min=0,
        help="Iterations run and discarded before the clock starts. The first "
        "pass through a fresh process compiles, autotunes and captures; "
        "measuring that publishes a compile time as a latency.",
    ),
]
Label = Annotated[str | None, typer.Option(help="Name for this run in the record.")]
Out = Annotated[
    str | None, typer.Option("-o", "--out", help="Write the JSON record here.")
]
BaseUrl = Annotated[str, typer.Option(help="Base URL of the running server.")]
ServedModel = Annotated[
    str | None,
    typer.Option(
        "--model",
        help="Model id to request. Defaults to whatever /v1/models advertises.",
    ),
]
NumPrompts = Annotated[int, typer.Option(min=1, help="Requests to send.")]
MaxConcurrency = Annotated[
    int | None,
    typer.Option(
        min=1,
        help="Requests allowed to execute at once. Models a bottleneck in "
        "front of the engine; leave it off and the arrival rate is the only "
        "limit. Set it no higher than the server's --max-batch-size unless "
        "queueing is the thing you are measuring.",
    ),
]
#: One flag for prompt length, generation length and their spread, because
#: those three are one decision. The record carries the name, so two runs can
#: be checked for comparability instead of trusted.
ShapeName = Annotated[
    str,
    typer.Option(
        "--shape",
        help="Workload shape — how much prompt against how much generation. "
        + "; ".join(f"{s.name} ({_sizes(s)}) {s.what}" for s in SHAPES.values())
        + ". This is the axis walnut is most sensitive to: prefill runs alone "
        "and unchunked, so a long prompt stalls every stream already running. "
        "A conclusion drawn at one shape does not transfer to another.",
    ),
]
Tokenizer = Annotated[
    str | None,
    typer.Option(
        help="Tokenizer used to build prompts for the generated shapes. "
        "Defaults to the model."
    ),
]
Goodput = Annotated[
    list[str] | None,
    typer.Option(
        "--goodput",
        metavar="KEY:MS",
        help="SLOs a request must meet to count, e.g. `--goodput ttft:250 "
        "--goodput tpot:10`. A request counts only if it cleared every one. "
        "Keys: " + ", ".join(SLO_METRICS) + ".",
    ),
]


def _shape(name: str):
    try:
        return resolve_shape(name)
    except BenchError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _fail(exc: BenchError) -> None:
    """Refusals print where the numbers print, then exit non-zero."""
    typer.echo(f"\n! {exc}")
    raise typer.Exit(1)


def _ladder(value: str) -> list[float]:
    """An ascending, comma-separated ladder of finite rates."""
    parsed = [float(v) for v in value.split(",") if v.strip()]
    if not parsed or any(r <= 0 or math.isinf(r) for r in parsed):
        raise typer.BadParameter("--rates takes positive, finite req/s values")
    if parsed != sorted(parsed):
        # The sweep stops at the first rung the server cannot absorb, which
        # only finds a knee if the ladder climbs.
        raise typer.BadParameter("--rates must ascend")
    return parsed


def _serve_options(**kwargs) -> ServeOptions:
    try:
        goodput = goodput_config(kwargs.pop("goodput"))
    except BenchError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return ServeOptions(goodput=goodput, shape=_shape(kwargs.pop("shape")), **kwargs)


# -- serve ------------------------------------------------------------------


@app.command()
def serve(
    base_url: BaseUrl = DEFAULT_BASE_URL,
    model: ServedModel = None,
    num_prompts: NumPrompts = 200,
    request_rate: Annotated[
        float,
        typer.Option(
            help="Requests per second, on a Poisson arrival process. Unset "
            "submits everything at once, which measures a saturated engine "
            "and says nothing about queueing.",
        ),
    ] = float("inf"),
    max_concurrency: MaxConcurrency = None,
    shape: ShapeName = DEFAULT_SHAPE,
    tokenizer: Tokenizer = None,
    seed: Annotated[
        int,
        typer.Option(
            help="Seeds the arrival process and the prompt generation, so a "
            "run reproduces end to end.",
        ),
    ] = 0,
    num_iters_warmup: NumItersWarmup = 1,
    goodput: Goodput = None,
    timeout: Annotated[float, typer.Option(help="Per request, seconds.")] = 600.0,
    ready_timeout: Annotated[float, typer.Option(help="Seconds.")] = 600.0,
    profile: Annotated[
        bool,
        typer.Option(
            help="Wrap the measured window in /start_profile and /stop_profile. "
            "Needs WALNUT_TORCH_PROFILER_DIR on the server. Profiled timings "
            "are inflated; take the trace from this run and the numbers from a "
            "clean one.",
        ),
    ] = False,
    label: Label = None,
    out: Out = None,
) -> None:
    """Drive a running `walnut serve` over HTTP, under an arrival process.

    The number a serving change is judged on, and the only one that sees
    queueing. Start the server first, sized for the load you intend to offer.

    Arrivals are Poisson, generation is greedy, and every request generates
    exactly the shape's output length. None of the three is a knob.
    """
    if request_rate <= 0:
        raise typer.BadParameter("--request-rate must be positive")
    opts = _serve_options(
        base_url=base_url,
        model=model,
        num_prompts=num_prompts,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        shape=shape,
        tokenizer=tokenizer,
        seed=seed,
        warmups=num_iters_warmup,
        goodput=goodput,
        timeout=timeout,
        ready_timeout=ready_timeout,
        label=label,
        out=out,
        profile=profile,
    )
    try:
        raise typer.Exit(asyncio.run(run_serve(opts)))
    except BenchError as exc:
        _fail(exc)


# -- sweep ------------------------------------------------------------------


@app.command()
def sweep(
    rates: Annotated[
        str,
        typer.Option(
            help="Ascending, comma-separated req/s ladder, e.g. `8,16,24,32`."
        ),
    ],
    base_url: BaseUrl = DEFAULT_BASE_URL,
    model: ServedModel = None,
    num_prompts: NumPrompts = 200,
    max_concurrency: MaxConcurrency = None,
    shape: ShapeName = DEFAULT_SHAPE,
    tokenizer: Tokenizer = None,
    seed: Annotated[int, typer.Option(help="Seeds the run end to end.")] = 0,
    num_iters_warmup: NumItersWarmup = 1,
    goodput: Goodput = None,
    goodput_floor: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Stop when the fraction of requests meeting --goodput falls "
            "below this. Ignored without --goodput.",
        ),
    ] = 0.95,
    timeout: Annotated[float, typer.Option(help="Per request, seconds.")] = 600.0,
    ready_timeout: Annotated[float, typer.Option(help="Seconds.")] = 600.0,
    label: Label = None,
    out: Out = None,
) -> None:
    """Run `serve` up a ladder of rates until the server stops keeping up.

    Stops at the first rung it cannot absorb — goodput through the floor, or
    the client behind its own schedule. Rungs above that measure a queue, not
    an engine. The rung it stops on is the operating point.
    """
    ladder = _ladder(rates)
    opts = _serve_options(
        base_url=base_url,
        model=model,
        num_prompts=num_prompts,
        request_rate=float("inf"),
        max_concurrency=max_concurrency,
        shape=shape,
        tokenizer=tokenizer,
        seed=seed,
        warmups=num_iters_warmup,
        goodput=goodput,
        timeout=timeout,
        ready_timeout=ready_timeout,
        label=label,
        out=out,
    )
    try:
        raise typer.Exit(asyncio.run(run_sweep(opts, ladder, goodput_floor)))
    except BenchError as exc:
        _fail(exc)


# -- latency ----------------------------------------------------------------


@app.command()
def latency(
    model: Model,
    shape: ShapeName = DEFAULT_SHAPE,
    num_iters: NumIters = 5,
    num_iters_warmup: NumItersWarmup = 3,
    temperature: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="0 (the default) is greedy, which is what makes the output "
            "hash a correctness check. It also never runs the sampler's softmax "
            "path — raise this, with --seed, if the sampler is what changed.",
        ),
    ] = 0.0,
    top_p: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Nucleus sampling p. Only meaningful above --temperature 0, "
            "and only measurable here: the sort it adds is visible with the "
            "scheduler and HTTP out of the way, and nowhere else.",
        ),
    ] = 1.0,
    seed: Annotated[int | None, typer.Option(help="Seed for sampled runs.")] = None,
    device: Device = "auto",
    dtype: Dtype = "auto",
    cuda_graph: CudaGraph = True,
    compile: Compile = True,
    autotune: Autotune = True,
    label: Label = None,
    out: Out = None,
) -> None:
    """Drive MODEL in-process, one stream, decode path only.

    Exactly one request is in flight — there is no batch-size knob, because a
    batch would put the scheduler back in the measurement. Greedy by default,
    with the output hashed so a numerics change cannot pass as a speedup.

    For batched numbers: `throughput` sizes a batch offline, `serve` gives
    per-request latency under one.
    """
    opts = EngineOptions(
        model=model,
        device=device,
        dtype=dtype,
        cuda_graph=cuda_graph,
        compile=compile,
        autotune=autotune,
        max_batch_size=1,
    )
    try:
        raise typer.Exit(
            run_latency(
                opts,
                shape=_shape(shape),
                num_iters=num_iters,
                num_iters_warmup=num_iters_warmup,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                label=label,
                out=out,
            )
        )
    except BenchError as exc:
        _fail(exc)


# -- throughput -------------------------------------------------------------


@app.command()
def throughput(
    model: Model,
    num_prompts: NumPrompts = 200,
    shape: ShapeName = DEFAULT_SHAPE,
    max_batch_size: Annotated[
        int,
        typer.Option(
            min=1,
            envvar="WALNUT_MAX_BATCH_SIZE",
            help="Requests decoded as one batch — the knob this sizes.",
        ),
    ] = 8,
    max_seq_len: Annotated[
        int | None, typer.Option(min=1, help="Context each batch slot holds.")
    ] = None,
    num_iters_warmup: NumItersWarmup = 1,
    seed: Annotated[int, typer.Option(help="Seeds prompt generation.")] = 0,
    device: Device = "auto",
    dtype: Dtype = "auto",
    cuda_graph: CudaGraph = True,
    compile: Compile = True,
    autotune: Autotune = True,
    label: Label = None,
    out: Out = None,
) -> None:
    """Submit every request to MODEL at once, in-process, and time the lot.

    No HTTP and no arrival process: the engine's ceiling, not what a client
    sees. Everything is offered at once, so there is no queueing to observe.

    Throughput only. Streams are drained one after another, so a token's read
    time is not its produce time and per-request latency here would be
    fiction; `serve` is where latency under a batch comes from.

    Greedy, and every request held to exactly the shape's output length.
    """
    opts = EngineOptions(
        model=model,
        device=device,
        dtype=dtype,
        cuda_graph=cuda_graph,
        compile=compile,
        autotune=autotune,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    try:
        raise typer.Exit(
            run_throughput(
                opts,
                shape=_shape(shape),
                num_prompts=num_prompts,
                num_iters_warmup=num_iters_warmup,
                seed=seed,
                label=label,
                out=out,
            )
        )
    except BenchError as exc:
        _fail(exc)


# -- startup ----------------------------------------------------------------


@app.command()
def startup(
    model: Model,
    num_iters: NumIters = 3,
    num_iters_warmup: NumItersWarmup = 1,
    max_batch_size: Annotated[
        int,
        typer.Option(
            min=1,
            envvar="WALNUT_MAX_BATCH_SIZE",
            help="Batch slots to prepare. One graph per power-of-two bucket, "
            "so this drives the capture cost.",
        ),
    ] = 8,
    max_seq_len: Annotated[
        int | None, typer.Option(min=1, help="Context each batch slot holds.")
    ] = None,
    device: Device = "auto",
    dtype: Dtype = "auto",
    cuda_graph: CudaGraph = True,
    compile: Compile = True,
    autotune: Autotune = True,
    label: Label = None,
    out: Out = None,
) -> None:
    """Measure what it costs to get MODEL ready to serve, and nothing else.

    Three phases: weights off disk, compile-and-capture, then whatever the
    first request still has to do. Each iteration builds a whole engine and
    throws it away, so --num-iters-warmup absorbs the cold compile.

    The first request is always timed. It should be small; if it is not,
    something `start` ought to have done is being deferred into a request, and
    that is a diagnostic there is no reason to be able to switch off.
    """
    opts = EngineOptions(
        model=model,
        device=device,
        dtype=dtype,
        cuda_graph=cuda_graph,
        compile=compile,
        autotune=autotune,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    try:
        raise typer.Exit(
            run_startup(
                opts,
                num_iters=num_iters,
                num_iters_warmup=num_iters_warmup,
                label=label,
                out=out,
            )
        )
    except BenchError as exc:
        _fail(exc)
