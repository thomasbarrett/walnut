---
name: analyze-trace
description: >-
  Analyzes PyTorch profiler traces written by `walnut profile` or the server's
  /start_profile endpoint, using Perfetto's trace_processor. Reports GPU idle
  time, kernel-launch overhead, and the hottest operators by self time. Use
  when looking at a trace, a profile, or a .trace.json.gz file, or when
  answering why inference is slow — "profile this", "analyze the trace", "why
  is decode slow", "is this launch-bound", "where is the GPU idle", "did
  --cuda-graph help", "what is the hot kernel".
---

# Analyzing a walnut trace

`walnut profile` writes a Chrome trace, normally read by eye at
<https://ui.perfetto.dev/>. `scripts/analyze_trace.py` loads it into Perfetto's
`trace_processor` — a SQL database over the same file — and runs the queries
that are easy to get wrong.

Run everything from the repo root, through the project environment:

```bash
uv run python .claude/skills/analyze-trace/scripts/analyze_trace.py <command> <trace>
```

Requires `uv sync --extra cpu --dev`. Perfetto downloads `trace_processor_shell`
on first use. Add `--json` to any command for machine-readable output.

Do not parse the trace file directly — it is tens of megabytes of gzipped JSON,
and the slice nesting is easy to misread.

## Finding a trace

`walnut profile` prints the path it wrote. Otherwise the traces are files:

```bash
ls -t profiles/*.trace.json.gz | head
```

The default directory is `./profiles`, or `$WALNUT_TORCH_PROFILER_DIR` if set.
Each trace has a `.summary.txt` beside it — torch's own `key_averages()` table.

## Commands

| Command | Reports |
| --- | --- |
| `overview` | Span, slice categories, threads, whether there is GPU work. |
| `top-ops` | Operators ranked by self time. `--category`, `--limit`. |
| `device` | GPU busy time per stream, largest idle gaps. `--min-gap-us`. |
| `launch` | Kernel-launch cost against kernel execution time. |
| `sql` | Any PerfettoSQL. Reads stdin, so quoting is not a hazard. |

## Diagnosing a slow run

Run `overview` first. If `device work: no`, it is a CPU trace and the `device`
and `launch` commands have nothing to report — that is the answer, not a
failure.

On a GPU trace:

1. `device` — if `busy_pct` is high, the kernels are the cost. Go to
   `top-ops --category kernel`.
2. If `busy_pct` is low with large `idle_gaps`, the GPU is starved. Go to
   `launch`.
3. `launch` — a high `launch_per_device_pct` means launch-bound, which is what
   `--cuda-graph` addresses. If it stays high with CUDA graphs enabled, capture
   is not covering the hot path. A large `synchronize_ms` is the opposite
   problem: the CPU blocked waiting on the GPU.
4. Otherwise the gap is CPU work between launches: `top-ops --category cpu_op`,
   then `python_function`.

Report the numbers the commands print, and name the trace file you read.

## Reading the numbers

- **Self vs inclusive time.** `top-ops` ranks by `self_us`, exclusive of
  children, so `aten::addmm` outranks the `aten::linear` wrapping it.
  `inclusive_us` counts a nested slice again in every ancestor, so the totals
  in `overview` exceed the trace duration by design.
- **`busy_pct` is per stream**, against the whole trace span. Streams run
  concurrently; adding them together can exceed 100%.
- **A server-side profile has no op names.** Kineto records `aten::` ops only
  on the thread that opened the window, so a trace from `/start_profile` shows
  the CUDA timeline without CPU op names. Use `walnut profile` for both — see
  `walnut/profiler.py`.

## Custom queries

**`ts` and `dur` are nanoseconds.** The commands above convert for you, raw SQL
does not, so divide by 1e6 for milliseconds — `dur / 1000` looks like ms and is
out by 1000×.

Pipe the query in; `--sql` also works but invites shell-quoting mistakes.

```bash
echo "select name, count(*) as n from slice where category = 'kernel'
      group by name order by n desc" |
  uv run python .claude/skills/analyze-trace/scripts/analyze_trace.py sql profiles/x.trace.json.gz
```

For the table layout, torch's category tags, and worked examples, see
[references/perfetto-schema.md](references/perfetto-schema.md).
