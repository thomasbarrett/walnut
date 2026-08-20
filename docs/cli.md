# CLI

Commands run as `uv run walnut <command>` (or `walnut <command>` inside an
activated virtualenv).

The reference below is generated directly from the Typer application, so it
always matches the installed version. You can also run `walnut --help` or
`walnut <command> --help` for the same information from your terminal.

::: mkdocs-typer2
    :module: walnut.cli
    :name: walnut

## Device and precision

**walnut serves on CUDA only, from Ampere (compute capability 8.0) onwards.**
It decodes through FlashAttention's variable-length kernel, which has no CPU
build and no float32 or pre-Ampere one. A CPU selection, an older card, or
`--dtype float32` is refused up front rather than allowed to fail inside the
first request. Running it needs the CUDA wheels: `uv sync --extra cu130`, not
`--extra cpu`, which installs a torch that can lint and test but not serve.

`--device` and `--dtype` both default to `auto`: the default CUDA device, at
the dtype the checkpoint declares.

```console
$ uv run walnut serve Qwen/Qwen3.5-0.8B --device cuda:1 --dtype bfloat16
```

`auto` adjusts the checkpoint's dtype twice: a float32 checkpoint is downcast to
`bfloat16`, and `bfloat16` falls back to `float16`, with a warning, on CUDA
devices that cannot do bf16.

## Batch size

`--max-batch-size` is how many requests the server decodes as one batch,
defaulting to 8. It is the same knob as vLLM's `--max-num-seqs` and SGLang's
`--max-running-requests`: the number of sequence slots, and so the ceiling on
concurrency.

```console
$ uv run walnut serve Qwen/Qwen3.5-0.8B --max-batch-size 16
```

The batch is continuous. A request joins at the next decode step rather than
waiting for the current group to finish, and a finished sequence frees its slot
the step it stops — so a slow request never holds a fast one behind it. Prefill
runs on its own, one request at a time, into the slot the request was given.

Batching trades a stream's own latency for the engine's throughput. On an RTX
5090 serving Qwen3.5-0.8B, eight streams at once cost each of them 63% more
time per output token than a stream running alone, and produce 4.1× the tokens
per second overall.

Each slot preallocates its own KV cache, so the memory a batch costs is
`--max-batch-size` times `--max-seq-len`, whether or not the requests ever
arrive. `--max-seq-len` is the context each slot is allocated for — prompt plus
completion — and defaults to the checkpoint's own limit capped at 8192; a
request that does not fit is rejected with a 400 rather than allowed to displace
a running one. Decode reads only as far as each sequence has actually got, so a
long `--max-seq-len` costs memory but not time.

`walnut profile` takes both flags, defaulting to a batch of 1: it traces one
generation, and a larger batch would fill the trace with padding rows.

## CUDA graphs

Decode is replayed from a captured CUDA graph by default, which removes the
per-token kernel launch cost. One graph is captured per power-of-two batch size
and the smallest one that fits is replayed, so the capture count grows with the
logarithm of `--max-batch-size`. Capture happens at start-up, before the port
opens, so no request pays for it; `--no-cuda-graph` turns it off. The flag is
ignored off CUDA, and `walnut profile` takes it with the same default, so a
profile measures what a server runs.

```console
$ uv run walnut serve Qwen/Qwen3.5-0.8B --no-cuda-graph
```

## Compilation

The decode step also runs through `torch.compile` by default, which fuses the
elementwise chains the norms and the delta-rule recurrence would otherwise
spend a kernel apiece on. Prefill stays eager: its shapes follow the prompt, so
compiling it would recompile per prompt length, while decode's are fixed and
one compile serves every request.

That compile costs a few seconds, paid once on the first request; `--no-compile`
turns it off. Like `--cuda-graph`, `walnut profile` takes the flag with the same
default.

```console
$ uv run walnut serve Qwen/Qwen3.5-0.8B --no-compile
```

## Autotuning

That compile also autotunes by default: for each projection Inductor benchmarks
a Triton kernel against cuBLAS and keeps whichever is faster, rather than taking
cuBLAS on faith. It is worth doing because decode's matmuls are matrix-*vector*
products at the batch sizes serving actually reaches — and cuBLAS's `gemv`
serves the narrow ones at roughly half the GPU's bandwidth. On an RTX 5090
picking per shape is worth ~11% of TPOT.

Autotuning runs at compile time, so it lands on the first request: expect
seconds rather than the fraction of a second `--compile` alone costs. Inductor
caches the result on disk beside the compiled graph, so the cost is one cold
compile per build, not one per process. `--no-autotune` turns it off, and it is
ignored with `--no-compile`, which skips the compile that would do the tuning.

```console
$ uv run walnut serve Qwen/Qwen3.5-0.8B --no-autotune
```

## Sampling while profiling

`walnut profile` decodes greedily by default (`--temperature 0`), which is what
`walnut bench` measures. Matching matters: at
`--temperature 1.0` the sampler's softmax over a 248k vocabulary is the largest
non-graph kernel in the trace, and a greedy run never executes it — so a trace
taken at a different temperature describes a different workload than the numbers
beside it.

Raise it when the sampler is what you are profiling.

```console
$ uv run walnut profile Qwen/Qwen3.5-0.8B --temperature 1.0
```

## Benchmarking

`walnut bench` has five subcommands, each answering a different question:

| | drives | answers |
|---|---|---|
| `serve` | a running server, over HTTP | what a client gets, under load |
| `throughput` | the engine in-process, all requests at once | the engine's ceiling |
| `latency` | the model in-process, one stream | what a kernel change moved |
| `startup` | engine construction, repeatedly | what a restart costs |
| `sweep` | `serve`, up a ladder of request rates or concurrency limits | where capacity runs out, or what an operating point costs |

`serve` and `sweep` drive a server you start yourself; the other three load the
model in-process and take the same `--device`, `--dtype`, `--cuda-graph`,
`--compile` and `--autotune` flags as `walnut serve`.

```console
$ uv run walnut serve Qwen/Qwen3.5-0.8B --max-batch-size 16 &
$ uv run walnut bench serve --shape chat --request-rate 16 --num-prompts 120 \
    --goodput ttft:250 --goodput tpot:10
```

`--shape` picks the workload — how much prompt against how much generation —
in one flag, because those are one decision. `chat` (the default, up to 1024 in
and exactly 1024 out), `rag`, `reasoning` and `agentic` span 1k to 16k prompt
tokens, with sizes taken from published benchmarks rather than invented. The
input figure is a ceiling: prompts are jittered over 80-100% of it, one-sided,
so a shape's name states a maximum a reader can check. That range matters
more here than in most engines: walnut prefills one request at a time, so a
long prompt still interrupts every stream already decoding — in chunks of
`--prefill-chunk` rather than all at once — and a conclusion drawn at one shape
does not transfer to another.

Every subcommand takes `--num-iters-warmup`, and every default is non-zero. The
first pass through a fresh process compiles the decode step, autotunes it and
captures a CUDA graph; measuring that iteration reports a compile time as a
latency. `walnut bench startup` is where that cost is the measurement rather
than the contaminant — each of its iterations builds a whole engine and
discards it, so the warm-ups absorb the cold compile and what remains is what a
restart pays.

`-o record.json` writes the full record, including the distributions the
printed table only digests.

## Environment variables

- `WALNUT_HOST` — default host for `walnut serve` (overridden by `--host`).
- `WALNUT_PORT` — default port for `walnut serve` (overridden by `--port`).
- `WALNUT_DEVICE` — default device for `walnut serve` and `walnut profile`
  (overridden by `--device`).
- `WALNUT_DTYPE` — default dtype for `walnut serve` and `walnut profile`
  (overridden by `--dtype`).
- `WALNUT_CUDA_GRAPH` — set to `0` to disable CUDA graph decode by default.
- `WALNUT_COMPILE` — set to `0` to disable `torch.compile` on decode by default.
