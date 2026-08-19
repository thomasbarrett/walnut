---
name: benchmark
description: >-
  Measure walnut's serving performance — TTFT, TPOT, ITL and end-to-end
  latency percentiles, request and token throughput, goodput against SLOs, and
  start-up cost. Use when asked how fast something is, to take a baseline
  before a change, to check a change for a regression, to size a batch or a
  concurrency limit, or to answer "is this faster", "what's the tok/s", "how
  many requests can it take", "did that help", "benchmark this".
---

# Benchmarking walnut

Numbers quoted to a human — or into a PR — come from here, not from a trace. A
profiled run inflates wall time badly; use `analyze-trace` for *why* a number
is what it is, and this for *what* it is.

## Pick the subcommand first

| `walnut bench` | drives | includes | answers |
|---|---|---|---|
| `serve` | a running server, over HTTP | queueing, batching, detokenization, HTTP | what a client gets, under load |
| `throughput` | the engine in-process, all at once | batching, no HTTP, no arrivals | the engine's ceiling |
| `latency` | the model in-process, one stream | the decode loop, nothing else | what a kernel change moved |
| `startup` | engine construction, repeatedly | weight load, compile, capture | what a restart costs |
| `sweep` | `serve`, up a ladder of rates | everything `serve` does, per rate | where capacity runs out |

**Changed a kernel, a fusion, the CUDA graph, the sampler? → `latency`.**
**Changed the scheduler, batching, the server, admission? → `serve`.**
**Sizing a batch? → `throughput`. Choosing an operating point? → `sweep`.**
Changed something that touches both → run both.

`serve` and `latency` are not substitutes. `serve` is what a serving change is
judged on and the only one that sees queueing, but it carries enough else that
a 3% decode win vanishes into it. `latency` makes that 3% visible and hashes
the output as a correctness check, but never notices a starved stream.

## Reading the table

Every subcommand with a distribution prints the same shape — one row per
metric, one column per statistic:

```
                           mean      p50      p90      p99      max       cv        n
TTFT (ms)                 19.46    19.72   19.87*   19.87*    19.87     2.2%        5
TPOT (ms)                  1.61     1.61    1.63*    1.63*     1.63     1.1%        5
ITL (ms)                   1.60     1.36     1.37    15.55    17.95        —      315
E2EL (ms)                120.73   121.02  122.58*  122.58*   122.58     1.2%        5
* rank equals the sample count: this is the maximum, not a tail.
```

**Read down a column, not across a row.** Whether the TTFT tail is blowing up
while TPOT holds is the question that matters, and it is one glance down `p99`.

**`cv`** is the standard deviation as a percentage of the median — the noise
floor. It appears only where the samples are repeats of one measurement
(`latency`, `startup`); on a workload of 200 differently-scheduled requests
dispersion describes the traffic, not the measurement, and the column is
dropped. **A change wants to clear roughly 2× `cv` before it means anything** —
one standard deviation covers about two thirds of a sample.

**`n`** is a column because it varies by row: ITL pools every gap of every
request and reaches thousands where per-request metrics have hundreds. A p99
means different things at those two sizes.

**`*`** marks a percentile whose nearest rank *is* the sample count — one
request's latency wearing a tail statistic's name. Five iterations cannot
support a p90; the star says so rather than letting the number pass.

**Percentiles are nearest-rank**, so every one printed is a latency something
actually saw.

## The metrics

| | definition | what moves it |
|---|---|---|
| **TTFT** | request sent → first content delta | prefill, queueing behind other prefills |
| **ITL** | gap between consecutive deltas, one sample each | the decode step — and every prefill that interrupted it |
| **TPOT** | `(e2el − TTFT) / (output_tokens − 1)` | decode, averaged over the request |
| **E2EL** | request sent → last content delta | the whole request |
| **NTPOT** | `e2el / output_tokens` | recorded, never printed |
| **output tok/s** | generated tokens / duration | the engine as a system |
| **concurrency** | `Σ e2el / duration` — Little's law | how full the engine was kept |
| **goodput** | requests/s meeting *every* SLO | the tail, which is what users leave over |

**TPOT is not `1/ITL`.** TPOT averages the decode phase per request; ITL is the
per-gap distribution. A prefill that stalls a running stream makes them diverge.

**NTPOT is recorded and never printed.** It divides whole-request latency by
token count, so a stalled prefill and a uniformly slow decode read the same.

**Output token counts come from the server's usage chunk**, never from counting
deltas. The detokenizer holds a piece back until it completes a character, so
one delta can carry two tokens; a delta count runs low and inflates TPOT
silently. `token_counts_exact: false` in the record means TPOT is an upper bound.

## Warm-up

Every subcommand takes `--num-iters-warmup`, and every default is non-zero. The
first pass through a fresh process compiles the decode step, autotunes it and
captures a CUDA graph. Folding that into a measured iteration publishes a
compile time as a latency; `--num-iters-warmup 0` prints a warning.

`startup` is where that cost is the subject rather than the contaminant.

## serve

```bash
walnut bench serve --num-prompts 120 --request-rate 16 --max-tokens 128 \
  --goodput ttft:250 --goodput tpot:10 -o rate16.json
```

Start the server first, sized for the load you intend to offer.

```
Successful requests:                 120
Concurrency (mean):                 8.07
Concurrency (peak):                   17
Request rate asked (req/s):        16.00
Request rate achieved (req/s):     13.79
Duration (s):                       9.10
Request throughput (req/s):        13.19
Request goodput (req/s):           13.19
Goodput (% of requests):           100.0
Output throughput (tok/s):       1688.65

                           mean      p50      p90      p99      max        n
TTFT (ms)                 27.56    24.17    41.35    56.01    62.34      120
TPOT (ms)                  4.60     4.88     5.84     6.38     6.40      120
ITL (ms)                   4.60     3.50     4.63    24.72    26.51    15240
E2EL (ms)                611.68   646.64   766.07   840.74   852.78      120
```

**Rate and concurrency are different knobs.** `--request-rate` is the traffic:
requests are submitted on a gamma arrival process (`--burstiness 1.0` is
Poisson) whether or not the server keeps up. This is open-loop, and the only
mode that builds a queue. Left off it fires everything at once, which measures
a saturated engine and says nothing about queueing.

`--max-concurrency` is a bottleneck *in front of* the engine. Set it no higher
than the server's `--max-batch-size` unless queueing is what you are measuring.
When set, a `queue wait (ms)` row appears — time spent in the client waiting
for a slot, held out of TTFT because it did not happen in the engine. A large
one means the server was offered less load than the rate implies.

**Read `Concurrency (mean)` and `(peak)` together.** The mean is what a result
quotes; the peak is the diagnostic. Below the limit you set, the arrival rate
was the constraint and the latencies describe a half-idle server. Above the
server's `--max-batch-size`, requests were queueing inside the engine.

**`--goodput ttft:250 --goodput tpot:10`** counts a request only if it cleared
*every* SLO. A request that answered in 80 ms then stalled for two seconds
served nobody, and each metric alone scores it a success. Keys: `ttft`, `tpot`,
`ntpot`, `itl`, `e2el`. An `itl` SLO is held against the request's **worst** gap.

**Datasets.** `--dataset fixed` (default) sends one prompt every time — cheap,
reproducible, right for a regression check. `--dataset random --input-len 512
--range-ratio 0.3` varies prompt length, which is what exercises a mixed batch.

**`--ignore-eos` is on by default** and holds every request to exactly
`--max-tokens`. A server that does not honour it is a hard error: ragged
lengths mean the latencies cannot be compared with each other.

**`--profile`** wraps the measured window in `/start_profile` and
`/stop_profile` (needs `WALNUT_TORCH_PROFILER_DIR` on the server) and prints
the trace path. Profiled timings are inflated — take the trace from that run
and the numbers from a clean one.

## throughput

```bash
walnut bench throughput Qwen/Qwen3.5-0.8B --num-prompts 200 --max-batch-size 16
```

Every request submitted at once, straight at the engine. No HTTP, no arrival
process, so no queueing to observe — this is the ceiling, and the gap between
it and `serve` at the same batch size is what the serving layer costs. On an
RTX 5090 with Qwen3.5-0.8B at `--max-batch-size 16`: **~3000 output tok/s**
here against ~1690 through `serve` at 16 req/s.

Throughput only, deliberately: streams are drained one after another, so a
token's read time is not its produce time and any per-request latency taken
here would be fiction. `serve` is where latency under a batch comes from.

## sweep

```bash
walnut bench sweep --rates 8,16,24,32 --num-prompts 200 \
  --goodput ttft:250 --goodput tpot:10 -o sweep.json
```

Runs `serve` once per rate and **stops at the first rung the server cannot
absorb** — goodput below `--goodput-floor` (0.95), or the client behind its own
arrival schedule.

```
  rate  achieved   conc  out tok/s  TTFT p99  TPOT p99  ITL p99  goodput
     8      7.68    2.8      975.6      48.9      4.87    21.86     100%
    16     15.02   10.5     1878.4     292.1      6.77    25.69      97%
    24     24.38   41.8     2328.2    2569.7     20.70    81.79      10%

stopped at 24 req/s: goodput fell to 10%, below the 95% floor. This is the knee.
```

The client kept up at every rung (24.38 offered against 24 asked), so this is
the engine's limit, not the harness's. **Throughput alone cannot tell you a
server is overloaded** — output tok/s is still climbing where goodput collapses.

One record for the whole ladder, so an operating point is chosen from a single
artifact. The shortfall stop compares time-to-submit against what the schedule
called for — deliberately not against `--request-rate`, since a finite gamma
sample has a realized mean of its own.

## latency

```bash
walnut bench latency Qwen/Qwen3.5-0.8B --label baseline -o before.json
```

128 output tokens, 3 warm-up iterations, 5 measured, greedy, driven straight at
the model — no scheduler, no HTTP, detokenization after the clock stops. Greedy
is what makes the output hash a correctness check; it also never runs the
sampler's softmax path, so raise `--temperature` (with `--seed`) if the sampler
is what you changed.

**Exactly one request is in flight, and there is no batch-size knob** — a batch
would put the scheduler back in the measurement, which is the thing this
excludes. Size a batch with `throughput`; get per-request latency under one
from `serve`.

**`tok/s` here is per-stream, not system throughput** — ~620 tok/s
single-stream against ~2290 tok/s at concurrency 16 on the same build. Never
quote one for the other.

**Read `Output sha` first.** Greedy sampling is deterministic, so a changed hash
means changed numerics and nothing else counts until that is explained. Every
iteration is hashed, so run-to-run nondeterminism is caught too — and exits
non-zero. It catches gross breakage, not the small numeric drift fusion work
causes: a relative 1e-4 perturbation leaves the hash identical.

## startup

```bash
walnut bench startup Qwen/Qwen3.5-0.8B --num-iters 3
```

```
                         median     mean       cv        n
load weights (s)           1.63     1.63     2.4%        3
prepare batch (s)          0.14     0.14     0.2%        3
first request (s)          0.03     0.03     0.4%        3
total (s)                  1.80     1.80     2.1%        3
```

Three phases, paid at different times and fixed by different work. Each
iteration builds a whole engine and throws it away, so the warm-ups absorb the
cold compile and what remains is what a restart pays.

A large `first request` means something `start` should have done up front is
being deferred into a request. `--no-first-request` drops that phase.

## Believing a delta

**Read the `cv` column, and want ~2× it.** On a quiet RTX 5090 at default
iteration counts, TTFT sits near 1.7% and TPOT near 0.5%, so a TPOT change
under ~1% has not been shown. `cv` converges by about ten iterations — raise
`--num-iters` if you need a tighter floor, and it will not drift for having
done so.

`serve` prints no `cv`, because its samples are not repeats. There the floor is
about 2% on medians and p99s at 120 requests, and under 0.1% on throughput.
Fewer requests, wider tails.

Two runs are only comparable if they measured the same thing — same model,
device, dtype, prompt, token count and flags. Nothing enforces that; check it.

## What this machine does

Closed loop (`--request-rate` unset), `--max-concurrency` swept:

```
conc  out tok/s  TPOT med  ITL p99  TTFT med  TTFT p99  E2EL med
   1      602.5      1.50     1.61      21.1      23.9     211.9
   2      885.4      2.05     2.29      32.1      42.2     288.6
   4     1369.4      2.57    22.27      42.1     214.8     367.1
   8     1980.4      3.73    23.72      43.0     152.9     513.3
  16     2286.0      6.59    26.56      45.5     339.8     885.1
```

Batching buys 3.8× system throughput for 4.4× each stream's TPOT. **The ITL p99
cliff between 2 and 4 is not noise** — median barely moves while p99 goes
2.3 → 22.3 ms. The cause is admission: a prefill stalls every running sequence
for its duration (`walnut/scheduler.py`, `Scheduler._admit`), so once requests
arrive mid-batch each running stream eats a ~20 ms gap. Anything reading only
the mean misses this.

## Which number leads

- **Decode kernels, fusion, graph contents** → `latency` TPOT, against its `cv`.
  ITL p99 tracks the mean there, so read `ITL max` for genuine spikes.
- **Scheduler, batching, admission** → `serve` output tok/s *and* ITL p99. A
  change that raises throughput while widening the ITL tail traded away what
  users feel.
- **Prefill, chunking, prompt handling** → `serve` TTFT p50 and p99, at a
  finite `--request-rate` so prefills contend.
- **Batch sizing** → `throughput` across `--max-batch-size`.
- **Capacity, "how many can it take"** → `sweep`. The rung it stops on is the
  answer.
- **Compilation, autotuning, lazy init** → `startup`.
- **TTFT alone never justifies a change.** `--no-cuda-graph` *improves* TTFT
  here by 3.7% while TPOT goes 1.48 → 3.37 ms and per-stream throughput falls
  56%. A run reporting only TTFT would have called that a win.

## Method

- **Baseline first**, before you edit.
- **Quiet machine, same machine.** GPU clocks drift with temperature, and the
  "after" run is always the one at the end of a long session.
- **Say what the workload was.** `tok/s` without the concurrency, token count
  and prompt length is not a result. Every record carries all of them.
- **Match the profile.** `walnut profile` takes `--temperature` and
  `--max-tokens`; set them to the benchmark's, or the trace describes a
  different workload than the numbers beside it.
