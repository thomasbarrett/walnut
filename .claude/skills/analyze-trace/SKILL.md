---
name: analyze-trace
description: >-
  Collect and analyze PyTorch profiler traces from `walnut profile` or the
  server's /start_profile endpoint, querying them with PerfettoSQL. Use when
  capturing a profile, reading a .trace.json.gz file, or answering why
  inference is slow — "profile this", "analyze the trace", "why is decode
  slow", "is this launch-bound", "where is the GPU idle", "did --cuda-graph
  help", "what is the hot kernel".
---

# Analyzing a walnut trace

## Collect a profile

```bash
uv run walnut profile Qwen/Qwen3.5-0.8B --max-tokens 32
```

It prints the two paths it wrote: `profiles/walnut-<ts>-<pid>.trace.json.gz`
and a `.summary.txt` beside it, which is torch's own `key_averages()` table.
Flags: `--prompt`, `--max-tokens`, `--device`, `--dtype`, `--output-dir` (or
`$WALNUT_TORCH_PROFILER_DIR`), and `--cuda-graph` / `--no-cuda-graph`.

The command warms up before opening the profiling window and synchronizes
before closing it, so the trace is neither distorted by a cold start nor
truncated with kernels still in flight.

To profile a running server instead:

```bash
WALNUT_TORCH_PROFILER_DIR=./profiles uv run walnut serve <model>
curl -X POST localhost:8000/start_profile
#   ... drive a few requests ...
curl -X POST localhost:8000/stop_profile
```

Kineto records `aten::` ops only on the thread that opened the window, and
walnut serves on worker threads, so a server trace has the CUDA timeline but no
operator names. Prefer `walnut profile` unless the question is about serving.

To find existing traces: `ls -t profiles/*.trace.json.gz | head`

## Query it

Never read a `.trace.json.gz` by hand — it is tens of megabytes of gzipped JSON
and the slice nesting is easy to misread. Load it into Perfetto's
`trace_processor` and use SQL:

```bash
cat .claude/skills/analyze-trace/scripts/prelude.sql my_query.sql |
  uv run python .claude/skills/analyze-trace/scripts/analyze_trace.py sql <trace>
```

The `sql` subcommand reads stdin, accepts multiple statements, and prints the
last result. `scripts/prelude.sql` defines the views the reference's queries
are written against — `ev`, `dev_op`, `api`, `phase`, `link`, `kfam`, `gap` —
so without it those queries fail with `no such table`. Requires
`uv sync --extra cpu --dev`; Perfetto downloads `trace_processor_shell` on
first use.

`analyze_trace.py` also has canned `overview`, `top-ops`, `device`, and
`launch` commands for a first look without writing SQL.

`ts` and `dur` are nanoseconds: divide by 1e3 for µs, 1e6 for ms.

## Analyze it

Read [references/README.md](references/README.md) and follow it into the
chapters. It is a textbook on using PerfettoSQL to find bottlenecks in
inference traces — the trace format, the view layer, a triage procedure that
identifies which bottleneck you have, and the host-side and device-side
branches it sends you down.

One walnut-specific difference from the book: walnut emits no
`record_function` scopes, so `user_annotation` is empty and the book's `phase`
table would have zero rows, silently emptying every per-token query.
`prelude.sql` reads the Python frames instead — `_prefill`, `_decode_step` (one
per token), and `DecodeGraph.capture` — which walnut names for the purpose.
Match those names rather than the line numbers beside them.
