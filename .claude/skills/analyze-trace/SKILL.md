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

## Get a trace

Look for one first: `ls -t profiles/*.trace.json.gz | head`. Before analyzing an
existing trace, read the `.summary.txt` beside it and establish which model and
flags produced it — every conclusion is conditional on them, and a comparison
question like "did `--cuda-graph` help" is unanswerable without them. If you
cannot establish them, capture a fresh trace rather than guessing:

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
operator names. It also has no Python frames, so `phase` comes back empty and
every per-token query in Chapters 3 and 6 returns nothing. Prefer
`walnut profile` unless the question is about serving.

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
`launch` commands for a first look without writing SQL. They are orientation,
not validation — the §2.3 preflight is still the gate before you conclude
anything.

`ts` and `dur` are nanoseconds: divide by 1e3 for µs, 1e6 for ms.

## Analyze it

The chapters under [references/](references/README.md) are a textbook on
finding bottlenecks in PyTorch inference traces. Read them on demand, not front
to back:

1. **Preflight** — run the five checks in §2.3 before trusting any number.
   Skipping it is how a truncated trace becomes a confident, wrong answer.
2. **Triage** — work through §3.3 in order, starting with §3.3.0 (is a graph
   replaying?). On the verdict, branch to Chapter 4 (host-bound) or Chapter 5
   (device-bound). §3.3.4 is the whole procedure on one screen.
3. Chapter 1 and §2.1–2.2 are lookups for when a query fails or a table comes
   back empty.

Kernel families map back to source in `walnut/layers/` (norms, attention, the
delta-rule recurrence) and `walnut/models/` (the `nn.Linear` projections behind
the gemv family). For the bandwidth roofline in §5.2 you need a parameter count
— from `model.safetensors.index.json` or the model card, not `config.json`,
which carries dims and `torch_dtype` only.

walnut emits no `record_function` scopes, so `prelude.sql` builds `phase` from
the Python frames instead — see its header for the frame names and why an empty
`phase` table is the symptom of losing them.

## Report it

A finished analysis states:

- **Where the token goes**, as an accounting identity that closes:
  `wall = GPU busy + idle`, measuring busy as the union of device intervals
  (§3.3.1) and idle as the gap total (§3.4.1), with the largest gaps attributed
  (§3.4.2). If it does not close, you have miscounted — find out why before
  writing anything down. Note §3.2.1's host-side split is a *different*
  decomposition whose terms overlap and whose residual closes by construction;
  it localizes cost, it does not verify it.
- **The ceiling and the headroom** — current tok/s against the GPU-side floor
  (§3.2.2) or the bandwidth roofline (§5.2), as a ratio.
- **Ranked findings**, each with its cost in µs/token or % of the phase, and
  the source file it lives in.

Before reporting, check the traps in §6.4 — at minimum the ones for the branch
you took. Two apply to almost every walnut trace: drop the warm-up token
(`WHERE seq > 0`), and do not sum kernel durations across streams.
