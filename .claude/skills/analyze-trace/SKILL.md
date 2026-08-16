---
name: analyze-trace
description: >-
  Collects and analyzes PyTorch profiler traces from `walnut profile` or the
  server's /start_profile endpoint, using PerfettoSQL over Perfetto's
  trace_processor. Use when capturing a profile, looking at a trace or a
  .trace.json.gz file, or answering why inference is slow — "profile this",
  "analyze the trace", "why is decode slow", "is this launch-bound", "where is
  the GPU idle", "did --cuda-graph help", "what is the hot kernel", "what is
  our TTFT / inter-token latency".
---

# Analyzing a walnut trace

Two halves: capture a trace worth trusting, then answer questions about it in
SQL. `references/` is a textbook on the second half — the trace format, a SQL
view layer, and 43 diagnostic queries. This file covers capture, the walnut
specifics the book cannot know, and which chapter to open.

Never read a `.trace.json.gz` by hand. It is tens of megabytes of gzipped JSON
and the slice nesting is easy to misread.

## 1. Capture

```bash
uv run walnut profile Qwen/Qwen3.5-0.8B --max-tokens 32
```

Writes `profiles/walnut-<ts>-<pid>.trace.json.gz` and a `.summary.txt` beside
it (torch's own `key_averages()` table) and prints both paths. Useful flags:
`--prompt`, `--max-tokens`, `--device`, `--dtype`, `--output-dir`
(or `$WALNUT_TORCH_PROFILER_DIR`), and `--cuda-graph` / `--no-cuda-graph`.

The command already does the two things people get wrong: it warms up with a
4-token generation before opening the window, and `TorchProfiler.close`
synchronizes before stopping so the tail of the GPU timeline is not truncated
(§2.1 explains why both matter). It captures with `record_shapes=True` and
`with_stack=True`, so shapes and `python_function` frames are present.

From a running server instead:

```bash
WALNUT_TORCH_PROFILER_DIR=./profiles uv run walnut serve <model>
curl -X POST localhost:8000/start_profile
#   ... drive a little traffic — a few requests, not a load test ...
curl -X POST localhost:8000/stop_profile
```

**A server trace has no `aten::` names.** Kineto records `cpu_op` only on the
thread that opened the window, and walnut serves on worker threads, so you get
the CUDA timeline with no operator names and no `aten::item` (which the phase
reconstruction below depends on). Prefer `walnut profile` for anything
diagnostic; use the endpoints when the question is specifically about serving.

Already have traces? `ls -t profiles/*.trace.json.gz | head`

## 2. Load

Every query in the book from Chapter 3 onward is written against a view layer —
`ev`, `dev_op`, `api`, `phase`, `link`, `kfam`, `gap`. `scripts/prelude.sql`
defines it. Pipe it in front of whatever you want to ask:

```bash
cat .claude/skills/analyze-trace/scripts/prelude.sql my_query.sql |
  uv run python .claude/skills/analyze-trace/scripts/analyze_trace.py sql <trace>
```

The `sql` subcommand takes multiple statements and prints the last result. It
reads stdin, so shell quoting is never a hazard. Requires
`uv sync --extra cpu --dev`; Perfetto downloads `trace_processor_shell` on
first use.

`analyze_trace.py` also has canned `overview`, `top-ops`, `device`, and
`launch` commands for a first look without writing SQL. They are a summary, not
the analysis — the book's queries answer questions those four cannot.

### The one place walnut departs from the book

walnut calls no `torch.profiler.record_function`, so `user_annotation` is
empty. The book's `phase` table would have **zero rows**, and every per-token
and per-phase query in Chapters 3 and 6 would silently return nothing.

`scripts/prelude.sql` therefore redefines `phase` from `aten::item`: walnut's
sampler pulls each token to the host with `int(next_token.item())` in
`iter_generate`, so there is exactly one per generated token. `prefill` runs
from trace start to the first token; `decode[i]` spans token *i* to token
*i+1*, so N tokens give N−1 decode intervals. Everything downstream — `link.ph`,
`link.seq`, per-token budgets, ITL percentiles — then works as written.

Two consequences to remember when reporting:

- The `prefill` phase also contains model warm-up and, with `--cuda-graph`,
  graph capture. It is not a clean TTFT measurement.
- If walnut ever wraps its phases in `record_function`, delete that block from
  the prelude — the book's own definition becomes correct.

## 3. Triage

Read §3.3 of
[Chapter 3](references/chapter-3-performance-model-and-triage.md) and run its
three queries. They decide which of the two branches you are on before you
propose a fix. In short: measure GPU busy fraction over a decode window, then
queue latency, then decompose host time.

Then follow the branch:

| Finding | Go to |
| --- | --- |
| GPU mostly idle, launches not keeping up | [Chapter 4](references/chapter-4-host-side-bottlenecks.md) — dispatch, syncs, CUDA graphs, `torch.compile` |
| GPU busy, the kernels themselves are the cost | [Chapter 5](references/chapter-5-device-side-bottlenecks.md) — kernel inventory, shapes, memory traffic, occupancy |
| Comparing two runs, or a tail-latency question | [Chapter 6](references/chapter-6-practice.md) — case study, ITL percentiles, A/B, CI |

Supporting material: [Chapter 1](references/chapter-1-trace-as-a-database.md)
for the table layout, the category and argument tables, and ingestion
semantics; [Chapter 2](references/chapter-2-capture-and-view-layer.md) for
capture detail and the preflight validation in §2.3.
[references/README.md](references/README.md) indexes the chapters and explains
the `§N.M` cross-references — the leading digit is the chapter.

**Run the preflight (§2.3) before concluding anything.** It catches dropped
events, a truncated GPU timeline, and unlinked device ops — all of which
produce confident, wrong numbers.

## 4. What a walnut decode trace looks like

Batch-1 decode multiplies a vector by each weight matrix, so every `aten::mm`
has M = 1 and the kernels are GEMVs a few microseconds long. That is the shape
of the workload, not a defect: decode goes launch-bound long before it goes
compute-bound, and the fixes are fewer launches, fused kernels, or a larger
batch — not a faster matmul. §5.2 covers the gemv trap.

Reference numbers, Qwen3.5-0.8B, 16 tokens on an RTX 5090, same prompt:

| | `--no-cuda-graph` | `--cuda-graph` |
| --- | --- | --- |
| decode wall time (15 intervals) | 419 ms | 70 ms |
| per token | 28.0 ms | 4.7 ms |
| **GPU busy during decode** | **13%** | **75%** |
| `cudaLaunchKernel` | 37,576 | 13,762 |
| `cudaGraphLaunch` | 0 | 16 (one per token) |
| max `link.launch_fanout` | 1 | 2,029 |
| decode device time | 54.9 ms | 53.1 ms |

The device work barely changed; the host stopped being the constraint. That is
the textbook launch-bound signature, and what §4.3 predicts CUDA graphs do.

When you compare two traces yourself, check they cover the same number of
decode steps first — `select count(*) from phase where name = 'decode'` — or
the totals are not comparable.

## 5. Reporting

Name the trace file you read, say which phase the numbers describe, and give
the busy fraction alongside any "GPU-bound" claim. Prefill and decode are
different workloads; a whole-trace average describes neither. §6.5 lists the
twelve mistakes that invalidate a result — check your answer against it before
you hand it over.
