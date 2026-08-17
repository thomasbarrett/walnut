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

`--device` and `--dtype` both default to `auto`: CUDA when it is available, at
the dtype the checkpoint declares. Running on CUDA needs the CUDA wheels
(`uv sync --extra cu130`, not `--extra cpu`).

```console
$ uv run walnut serve Qwen/Qwen3.5-0.8B --device cuda:1 --dtype bfloat16
```

`auto` adjusts the checkpoint's dtype twice: a float32 checkpoint is downcast to
`bfloat16` on an accelerator (but left alone on CPU), and `bfloat16` falls back
to `float16`, with a warning, on pre-Ampere CUDA devices.

## CUDA graphs

Decode is replayed from a captured CUDA graph by default, which removes the
per-token kernel launch cost. Capture happens once per request, after prefill,
and adds to time-to-first-token; `--no-cuda-graph` turns it off. The flag is
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
products — walnut serves one request at a time, so every projection is a batch
of one — and cuBLAS's `gemv` serves the narrow ones at roughly half the GPU's
bandwidth. On an RTX 5090 picking per shape is worth ~11% of TPOT.

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
the benchmark under `.claude/skills/benchmark/` measures. Matching matters: at
`--temperature 1.0` the sampler's softmax over a 248k vocabulary is the largest
non-graph kernel in the trace, and a greedy run never executes it — so a trace
taken at a different temperature describes a different workload than the numbers
beside it.

Raise it when the sampler is what you are profiling.

```console
$ uv run walnut profile Qwen/Qwen3.5-0.8B --temperature 1.0
```

## Environment variables

- `WALNUT_HOST` — default host for `walnut serve` (overridden by `--host`).
- `WALNUT_PORT` — default port for `walnut serve` (overridden by `--port`).
- `WALNUT_DEVICE` — default device for `walnut serve` and `walnut profile`
  (overridden by `--device`).
- `WALNUT_DTYPE` — default dtype for `walnut serve` and `walnut profile`
  (overridden by `--dtype`).
- `WALNUT_CUDA_GRAPH` — set to `0` to disable CUDA graph decode by default.
- `WALNUT_COMPILE` — set to `0` to disable `torch.compile` on decode by default.
