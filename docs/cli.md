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
ignored when serving off CUDA.

```console
$ uv run walnut serve Qwen/Qwen3.5-0.8B --no-cuda-graph
```

## Environment variables

- `WALNUT_HOST` — default host for `walnut serve` (overridden by `--host`).
- `WALNUT_PORT` — default port for `walnut serve` (overridden by `--port`).
- `WALNUT_DEVICE` — default device for `walnut serve` (overridden by `--device`).
- `WALNUT_DTYPE` — default dtype for `walnut serve` (overridden by `--dtype`).
- `WALNUT_CUDA_GRAPH` — set to `0` to disable CUDA graph decode by default.
