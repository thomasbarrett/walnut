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
profiled run inflates wall time badly (ITL 2.2 ms profiled vs 1.8 ms real, and
GPU-idle time inflates ~10×); the trace explains *why* a number is what it is,
and this measures *what* it is. Use `analyze-trace` for the why.

## The metrics

| | definition | what it covers |
|---|---|---|
| **TTFT** | request start → first token | prefill, CUDA graph capture, first sample |
| **ITL** | gap between consecutive tokens, one sample each | the decode step, per token |
| **TPOT** | `(e2e − TTFT) / (tokens_out − 1)` | decode, averaged over the request |
| **e2e** | request start → last token | the whole request |
| **output tok/s** | `1000 / TPOT` | per-stream decode rate |

Timestamps come from the engine's token stream (`iter_generate`), one per
token, with detokenization after the clock stops. `Engine.stream` is the wrong
thing to time: it yields per *decoded chunk*, so a multi-token character halves
the sample count and doubles apparent TPOT, and it re-detokenizes the whole
sequence every step — an O(n²) Python cost that shows up as ~4% ITL drift over
1024 tokens and is not the engine. That detokenizer is a real serving cost;
it is just not a decode cost, and conflating them hides both.

**These are single-stream numbers.** walnut serves one request at a time, so
there is no request rate, concurrency or goodput to sweep, and `tok/s` is
per-stream output throughput, not system throughput. When walnut gains
continuous batching this skill needs a load generator and the sweep that goes
with it.

**`tok/s` is a function of `--tokens`** — 553 at 128, ~499 at 512, ~447 at 1024,
because the KV cache is sized `prompt + max_new_tokens` and attention runs over
the whole static window. Never quote it without the token count.

## Take a baseline

```bash
BENCH=.claude/skills/benchmark/scripts/bench.py
uv run python $BENCH run Qwen/Qwen3.5-0.8B --label baseline -o /tmp/before.json
```

128 output tokens, 1 warm-up request, 5 measured, **greedy** (`--temperature 0`).
Greedy is what makes the output hash a correctness check. It also means the
sampler's softmax/multinomial path is never executed — the server defaults to
`temperature=1.0` and 256 tokens (`server.py:121-122`), so a bare run does *not*
measure a default serve request. Raise `--temperature` if the sampler is what
you're changing, and pass `--seed` to keep it reproducible.

Other flags: `--prompt`, `--tokens`, `--repeats`, `--warmup`, `--device`,
`--dtype`, `--cuda-graph`/`--no-cuda-graph`, `--compile`/`--no-compile`.

Give each record a distinct name and keep it — `compare` is only as good as the
files you still have.

## Establish the noise floor

Run it **twice on the same build**, as separate processes, before believing any
delta. Process-to-process variance exceeds the in-process `--repeats` spread.

```bash
uv run python $BENCH run <model> --label baseline-2 -o /tmp/before2.json
uv run python $BENCH compare /tmp/before.json /tmp/before2.json
```

```
                                before           after    change
TTFT (ms)                  51.89 ±2.6%     52.84 ±3.6%     +1.8%
TPOT (ms/token)             1.81 ±0.1%      1.81 ±0.2%     +0.1%
output tok/s                    552.89          552.41     -0.1%
ITL p99 (ms)                      1.82            1.83     +0.4%
e2e (ms)                  281.46 ±0.5%    282.59 ±0.6%     +0.4%
first request (ms)             2660.25         2649.03     -0.4%
```

`±` is the widest deviation from the median across repeats, carried next to the
value so neither is read without the other: TTFT moving +1.8% against ±2.6%
dispersion is nothing, while TPOT's ±0.1% means a 1% TPOT change is real. On a quiet RTX 5090 the per-token
metrics are stable to a tenth of a percent and the per-request ones to a few
percent. On an unmeasured machine, treat anything under ~5% as unproven.

## Compare

```bash
uv run python $BENCH compare /tmp/before.json /tmp/after.json
```

`compare` **exits non-zero and refuses to print a speedup** when the records
differ in model, device, GPU, dtype, prompt, token count, temperature, torch
version, or `flags`. Flipping `--compile` and calling the difference a speedup is
the easiest way to fake a result; the guard exists to make that impossible
rather than merely discouraged.

**Read the `output:` line first.** Greedy sampling makes it deterministic, so a
changed hash means changed numerics, and nothing below it counts until that is
explained. `compare` prints the first *differing* window, because divergence is
usually hundreds of characters in. Every repeat is hashed, so run-to-run
nondeterminism is caught too.

Know what this check does *not* cover: perturbing every weight by a relative
1e-4 leaves the hash identical, and at 1e-3 the outputs first differ around
character 480. It catches gross breakage, not the small numeric drift that
fusion work actually causes, and it exercises one prompt at one temperature.

### Explaining a changed hash

"Explained" means naming the mechanism, not asserting the change is exact. Each
step removes a layer, so stop at the first one that answers it.

1. **The op alone**, old vs new on the same input at the decode shape,
   `torch.equal`. Differing here means the change is wrong. Identical here
   points the drift at the shapes you didn't test — prefill dispatches
   differently.
2. **`--no-compile` on both sides.** Divergence survives ⇒ not Inductor.
3. **`--dtype float32` on both sides.** Identical ⇒ no difference large enough
   for fp32 `argmax` to see, i.e. bf16 rounding, usually cuBLAS choosing a
   different kernel for the new shape. Enough of an explanation for a PR.
4. **Still differing in float32** ⇒ a real numerics change. Explain it or
   revert.

Greedy `argmax` only flips at a near-tie, so rounding diverges *late* — in #40,
character 433. An immediate divergence is evidence against rounding.

## Which number leads

- **Decode path** (kernels, fusion, the graph's contents) → TPOT. ITL p99 tracks
  the mean closely here, so read `ITL max` for genuine spikes.
- **Prefill, graph capture, model setup** → TTFT. Note ~40% of TTFT is
  per-request CUDA graph capture, so `--no-cuda-graph` shows a *lower* TTFT
  alongside a 2.6× TPOT regression. TTFT alone never justifies a change.
- **Compilation, autotuning, lazy init** → `first request (ms)`. Weight loading
  is not in it; that happens before the timer.
- **Anything end to end** → e2e, but only alongside `tokens_out`.

## Method

- **Baseline first**, before you edit. A baseline reconstructed afterwards costs
  more and is worth less.
- **Warm up, then repeat.** The first request pays for compilation and lazy
  init. Per-request metrics are medians with the spread printed; ITL is pooled
  across repeats.
- **Fixed output length.** walnut has no `ignore_eos`, so a run can stop early;
  the record carries `tokens_out` and `hit_eos` and `run` warns on stderr.
  Per-token metrics survive a short run, e2e does not.
- **Quiet machine, same machine.** GPU clocks drift with temperature, so a run
  straight after a long profiling session is not comparable to a cold one — and
  the "after" run is always the one at the end of a long session. Check the
  printed spread.
- **One prompt is one workload.** Prompt length moves prefill, cache size and
  attention shape. If that's plausibly relevant, run `--prompt` twice.
- **Match the profile.** `walnut profile` takes `--temperature` and
  `--max-tokens`; set them to the benchmark's values or the trace describes a
  different workload than the one you measured.
