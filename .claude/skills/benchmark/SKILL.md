---
name: benchmark
description: >-
  Measure walnut's serving latency and throughput — TTFT, TPOT, ITL
  percentiles, end-to-end latency, output tok/s — and compare two runs for a
  regression. Use when asked how fast something is, to take a baseline before a
  change, to check a change for a regression, or to answer "is this faster",
  "what's the tok/s", "did that help", "benchmark this".
---

# Benchmarking walnut

Numbers quoted to a human — or into a PR — come from here, not from a trace. A
profiled run inflates wall time by 10–20%; the trace explains *why* a number is
what it is, and this measures *what* it is. Use `analyze-trace` for the why.

## The metrics

The standard serving set, measured client-side off `Engine.stream()`, the way a
user experiences them. Tools disagree on these definitions, so the formulas are
stated rather than assumed:

| | definition | what it covers |
|---|---|---|
| **TTFT** | request start → first token | prefill, CUDA graph capture, first sample |
| **ITL** | gap between consecutive tokens, one sample each | the decode step, per token |
| **TPOT** | `(e2e − TTFT) / (tokens_out − 1)` | decode, averaged over the request |
| **e2e** | request start → last token | the whole request |
| **output tok/s** | `1000 / TPOT` | per-stream decode rate |

TPOT and ITL measure the same thing from different ends: TPOT is the mean, ITL
carries the distribution. **Report ITL percentiles when a change could affect
jitter** — a graph capture, a sync, an allocation on the decode path — because a
p99 spike is invisible in a mean. Otherwise TPOT alone is fine.

**These are single-stream numbers.** walnut serves one request at a time, so
there is no request rate, concurrency, or goodput to sweep, and `tok/s` is
per-stream output throughput, not system throughput. When walnut gains
continuous batching, this skill needs a load generator and the sweep that goes
with it; until then, do not quote these as "throughput" without the qualifier.

## Take a baseline

```bash
BENCH=.claude/skills/benchmark/scripts/bench.py
uv run python $BENCH run Qwen/Qwen3.5-0.8B --label baseline -o /tmp/before.json
```

Defaults match the CLI's, so a bare run measures what `walnut serve` serves:
128 output tokens, 1 warm-up request, 5 measured. Flags: `--prompt`, `--tokens`,
`--repeats`, `--warmup`, `--device`, `--dtype`, and the two that change what is
being measured — `--cuda-graph`/`--no-cuda-graph`, `--compile`/`--no-compile`.

Keep the JSON. A benchmark you cannot diff later is half a benchmark.

## Compare

```bash
uv run python $BENCH compare /tmp/before.json /tmp/after.json
```

```
before: eager decode  {'cuda_graph': True, 'compile': False}
after:  compiled decode  {'cuda_graph': True, 'compile': True}
config: Qwen/Qwen3.5-0.8B on NVIDIA GeForce RTX 5090, bfloat16, 7 prompt tokens
        -> 128 output tokens, median of 5 runs

                           before      after     change
TTFT (ms)                   84.18      52.42     -37.7%
TPOT (ms/token)              3.56       1.82     -48.7%
output tok/s               280.91     547.97     +95.1%
ITL mean (ms)                3.53       1.81     -48.7%
ITL p99 (ms)                 3.56       1.83     -48.6%
e2e (ms)                   536.63     283.99     -47.1%
first request (ms)         689.23    2677.22    +288.4%

output: identical (61db643e7e0d)

speedup: 1.95x on TPOT
```

**Read `output:` first.** Sampling is greedy, so the hash is deterministic: if
it changed, the change altered what the model produces, and no speedup below it
means anything until that is explained. This is the accuracy half of the
benchmark, and it is not optional.

`compare` also warns when the two records disagree on model, device, GPU,
prompt, dtype or token count. Those runs are not comparable and the warning is
not advisory.

## Which number leads

Match the headline to what changed. A change that improves one metric at
another's expense is a trade, and reporting only the favourable half is how
benchmarks lose their credibility.

- **Decode path** (kernels, fusion, the graph's contents) → TPOT, with ITL p99
  if jitter is plausible.
- **Prefill, graph capture, model setup** → TTFT.
- **Compilation, autotuning, weight loading** → `first request (ms)`, which
  isolates the once-per-process cost. `torch.compile` shows up here as seconds;
  report it rather than hiding it behind a warm-up.
- **Anything user-facing end to end** → e2e, but only alongside `tokens_out`:
  e2e across runs that generated different token counts is meaningless.

## Method

- **Baseline first.** Measure before you edit. A baseline reconstructed
  afterwards by stashing changes costs more than the two minutes it would have
  taken, and is worth less.
- **Warm up, then repeat.** The first request pays for compilation, autotuning
  and lazy init; `--warmup` discards requests after it, `--repeats` measures.
  Per-request metrics are reported as medians and ITL as percentiles, because
  latency noise is one-sided — it only ever adds.
- **Fixed output length.** walnut has no `ignore_eos`, so a run can stop early;
  the record carries `tokens_out` and `hit_eos`, and `run` warns on stderr.
  Per-token metrics survive a short run, e2e does not. Pick a prompt that
  reliably generates the full `--tokens`.
- **Quiet machine, same machine.** GPU clocks drift with temperature, so a run
  straight after a long profiling session is not comparable to a cold one.
  Check `ttft_all_ms` and `e2e_all_ms` in the record: if the repeats disagree by
  more than a percent or two, something else is using the GPU.
- **Measure the noise floor; don't assume one.** Two runs of the *same* build,
  as separate processes, is the cheapest way to learn what a delta has to beat.
  On a quiet RTX 5090 that comes out under 0.2% on TPOT and e2e, ~0.6% on ITL
  p99, and ~1.5% on the first request — so the once-per-process metrics are the
  noisy ones. On an unmeasured machine, treat anything under ~5% as unproven.
- **One prompt is one workload.** A change can help a 7-token prompt and hurt a
  3000-token one — prompt length moves prefill, KV-cache size and attention
  shape. If that is plausibly relevant, run `--prompt` twice rather than
  generalizing from one.
- **Benchmark what people run.** The defaults are the defaults for a reason. If
  a flag combination is the point, say so in `--label`, which `compare` prints.
