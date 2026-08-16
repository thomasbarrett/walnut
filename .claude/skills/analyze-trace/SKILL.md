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

Every conclusion is conditional on the model and flags that produced the trace,
and **nothing in the trace or the `.summary.txt` records them** — the summary is
torch's `key_averages()` table, nothing more. So an existing file in `profiles/`
is only usable if you know how it was made. When in doubt, capture your own:

```bash
uv run walnut profile Qwen/Qwen3.5-0.8B --max-tokens 32
```

It prints the two paths it wrote: `profiles/walnut-<ts>-<pid>.trace.json.gz`
and a `.summary.txt` beside it. Flags: `--prompt`, `--max-tokens`,
`--temperature`, `--device`, `--dtype`, `--output-dir` (or
`$WALNUT_TORCH_PROFILER_DIR`), `--cuda-graph` / `--no-cuda-graph`, and
`--compile` / `--no-compile`.

**Match the workload you are reasoning about.** `--temperature` defaults to 0
(greedy), matching the `benchmark` skill; at `--temperature 1.0` the sampler's
softmax over a 248k vocabulary becomes the largest non-graph kernel in the
trace, and a greedy run never executes it. Likewise `--max-tokens` sets the KV
cache size, so it changes attention's cost: profile the length you benchmarked.

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
to back — each is one file, linked here directly:

- [Ch. 1 — the trace as a database](references/chapter-1-trace-as-a-database.md): event format, ingestion, tooling.
- [Ch. 2 — capture and the view layer](references/chapter-2-capture-and-view-layer.md): capture flags, the seven views, **§2.3 preflight**.
- [Ch. 3 — performance model and triage](references/chapter-3-performance-model-and-triage.md): prefill vs decode, per-token budget, **§3.3 triage**, §3.4 gap attribution.
- [Ch. 4 — host-side bottlenecks](references/chapter-4-host-side-bottlenecks.md): dispatch cost, syncs, CUDA graphs, `torch.compile`.
- [Ch. 5 — device-side bottlenecks](references/chapter-5-device-side-bottlenecks.md): kernel inventory, **§5.2 shapes and the gemv trap**, bandwidth, occupancy.
- [Ch. 6 — practice](references/chapter-6-practice.md): worked case study, tail latency, A/B comparison, **§6.4 twelve traps**.

The procedure:

1. **Preflight** — run the five checks in §2.3 before trusting any number.
   Skipping it is how a truncated trace becomes a confident, wrong answer.
2. **Triage** — work through §3.3 in order, starting with §3.3.0 (is a graph
   replaying?). On the verdict, branch to Chapter 4 (host-bound) or Chapter 5
   (device-bound). §3.3.4 is the whole procedure on one screen.
3. Chapter 1 and §2.1–2.2 are lookups for when a query fails or a table comes
   back empty.

Getting from a kernel to a `file:line` takes source reading, and two defaults
make it harder than the chapters assume:

- **Under a replayed CUDA graph there are no ATen ops**, so every kernel's `op`
  is `<built-in method replay>` and the decode phase is one opaque row.
  §5.2.0 recovers the shapes from the `cuda_graph_capture` phase, which runs
  eagerly. Without it you cannot get past "gemv is 73% of decode".
- **Under `torch.compile`, fused kernels are named for the ops they replaced**
  (`triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` is an RMS norm). §4.4.

With shapes in hand, the projections live in `walnut/models/qwen3_5.py` and
`walnut/layers/linear_attention.py`; norms, attention and the delta-rule
recurrence are in `walnut/layers/`.

For the bandwidth roofline in §5.2 you need the bytes the *decode path* reads —
from `model.safetensors.index.json` or the safetensors header, not
`config.json`, which carries dims and `torch_dtype` only. Exclude weights decode
never touches: on Qwen3.5-0.8B the vision tower (~201 MB) and the MTP head
(~41 MB) are 16% of the checkpoint, and counting them inflates the ceiling
enough to make a kernel already at 91% of peak look like it has headroom.

walnut emits no `record_function` scopes, so `prelude.sql` builds `phase` from
the Python frames instead — see its header for the frame names and why an empty
`phase` table is the symptom of losing them.

## Report it

A finished analysis states:

- **Where the token goes**, as an accounting identity that closes:
  `wall = GPU busy + idle`, measuring busy as the union of device intervals
  (§3.3.1) and idle as the gap total (§3.4.1), with the largest gaps attributed
  (§3.4.2). Filter both sides the same way — `gap` carries `seq` for exactly
  this. Expect the sum to fall ~1% *short* of wall: each phase begins and ends
  with host work that is neither busy nor an inter-kernel gap. A gap larger than
  a few percent, or a sum that *exceeds* wall, means you have miscounted —
  usually by summing kernels across streams. Note §3.2.1's host-side split is a *different*
  decomposition whose terms overlap and whose residual closes by construction;
  it localizes cost, it does not verify it.
- **The ceiling and the headroom** — current tok/s against the GPU-side floor
  (§3.2.2) or the bandwidth roofline (§5.2), as a ratio.
- **Ranked findings**, each with its cost in µs/token or % of the phase, and
  the source file it lives in.

Before reporting, check the traps in §6.4 — at minimum the ones for the branch
you took. Two apply to almost every walnut trace: drop the warm-up token
(`WHERE seq > 0`), and do not sum kernel durations across streams.
