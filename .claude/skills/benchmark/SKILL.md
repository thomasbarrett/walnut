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
| `sweep` | `serve`, up a ladder | everything `serve` does, per rung | where capacity runs out, or what an operating point costs |

**Changed a kernel, a fusion, the CUDA graph, the sampler? → `latency`.**
**Changed the scheduler, batching, the server, admission? → `serve`.**
**Sizing a batch? → `throughput`. Choosing an operating point? → `sweep`.**
Changed something that touches both → run both.

`serve` and `latency` are not substitutes. `serve` is what a serving change is
judged on and the only one that sees queueing, but it carries enough else that
a 3% decode win vanishes into it. `latency` makes that 3% visible and hashes
the output as a correctness check, but never notices a starved stream.

## Pick the shape second

`--shape` is one flag for prompt length, generation length and their spread,
because those are one decision. It is on `serve`, `sweep`, `throughput` and
`latency`, and the record carries the name — so two runs can be checked for
comparability instead of trusted.

| `--shape` | in / out | source | what it is |
|---|---|---|---|
| `chat` (default) | ≤1024 / 1024 | InferenceMAX | a conversational turn |
| `rag` | ≤4096 / 256 | Luminal | retrieval: prefill-heavy, short answer |
| `reasoning` | ≤1024 / 8192 | InferenceMAX | a long scratchpad, decode-bound |
| `agentic` | ≤16384 / 256 | Luminal | full history and tool schemas; prefill is the cost |

**The input figure is a ceiling, not a mean.** Prompts are jittered over
80–100% of it, one-sided, following InferenceMAX — so "`chat` sends at most
1024 prompt tokens" is a claim a reader can check against a record, where "about
1024 on average" is not. Output length is exact.

Every size is lifted from a published benchmark rather than invented, so a
walnut number can be read beside the source it came from. **There is no
industry standard** — no single publisher uses this exact set, and MLPerf, the
closest thing to an official one, samples lengths from real datasets instead of
pinning synthetic shapes. What the field agrees on is looser: ~1k input for
chat, one long-output case for reasoning, short outputs on the prefill-heavy
shapes so generation cost cannot mask prompt cost, and a little jitter on input
so a batch is mixed rather than uniform.

**This is the axis walnut is most sensitive to.** `Scheduler._admit` runs
prefill alone, one request at a time, unchunked — so a prompt's cost is paid by
every stream already decoding. Between `reasoning` and `agentic` the
prefill:decode work ratio moves by two orders of magnitude, and **no conclusion
drawn at one shape transfers to another**.

**`chat` is the default because it is the cheapest shape anyone actually
runs.** There is no smaller one worth having: a short synthetic prompt is
nearly all decode, and a regression gate that only guards decode passes changes
that ruin prefill.

The generated shapes build prompts from random token ids, which share no
prefix. That is deliberate today and will need revisiting the moment walnut
caches prefixes: multi-turn chat is nearly all prefix hits, and a no-hit
workload would understate it systematically.

The long shapes need room. `agentic` wants `--max-seq-len` past 16k on the
in-process subcommands, and a server started with enough context.

Prompts are generated from token ids, so every shape needs a tokenizer —
`--tokenizer`, or the model.

## Reading the table

Every subcommand with a distribution prints the same shape — one row per
metric, one column per statistic:

```
                           mean      p50      p90      p99      max      std        n
TTFT (ms)                 19.52    19.82   19.93*   19.93*    19.93    0.422        5
TPOT (ms)                  1.61     1.61    1.62*    1.62*     1.62    0.011        5
ITL (ms)                   1.60     1.36     1.37    15.72    17.57    1.898      315
E2EL (ms)                120.77   121.24  122.19*  122.19*   122.19    1.084        5
* rank equals the sample count: this is the maximum, not a tail.
```

**Read down a column, not across a row.** Whether the TTFT tail is blowing up
while TPOT holds is the question that matters, and it is one glance down `p99`.

**`std`** is the standard deviation, in the row's own units, and what it means
depends on what the samples are. On `latency` and `startup` they are repeats of
one measurement, so it is the noise floor: **a change wants to clear about 2×
`std` before it means anything**, since one standard deviation covers about two
thirds of a sample. On `serve` the samples are 120 differently-scheduled
requests, so it describes the traffic rather than the measurement, and the
percentiles are what to read.

It is a standard deviation and not the widest deviation from the median because
the latter is an extreme-value statistic: it climbs with sample count and never
settles. Measured here, max-deviation on TTFT went 1.9% → 4.8% between 5 and 40
iterations while `std` held — so two runs at different `--num-iters` could not
otherwise be read against each other.

**`n`** is a column because it varies by row: ITL pools every gap of every
request and reaches thousands where per-request metrics have hundreds. A p99
means different things at those two sizes.

**`*`** marks a percentile whose nearest rank *is* the sample count — one
request's latency wearing a tail statistic's name. Five iterations cannot
support a p90; the star says so rather than letting the number pass.

**Percentiles are nearest-rank**, so every one printed is a latency something
actually saw. **p50/p90/p99, fixed** — not a flag. A run reported at other
percentiles is comparable with no other run, and a knob here would hand out a
way around the `*` marker: p99.9 of five samples is the maximum, every time.

## The metrics

| | definition | what moves it |
|---|---|---|
| **TTFT** | request sent → first content delta | prefill, queueing behind other prefills |
| **ITL** | gap between consecutive deltas, one sample each | the decode step — and every prefill that interrupted it |
| **TPOT** | `(e2el − TTFT) / (output_tokens − 1)` | decode, averaged over the request |
| **E2EL** | request sent → last content delta | the whole request |
| **output tok/s** | generated tokens / duration | the engine as a system |
| **concurrency** | `Σ e2el / duration` — Little's law | how full the engine was kept |
| **goodput** | requests/s meeting *every* SLO | the tail, which is what users leave over |

**TPOT is not `1/ITL`.** TPOT averages the decode phase per request; ITL is the
per-gap distribution. A prefill that stalls a running stream makes them diverge.

**E2EL earns its row as a tail statistic and nothing else.** Per request it is
arithmetic — `e2el = ttft + tpot × (tokens − 1)`, by the definition of TPOT,
with output length held fixed — so the mean can never disagree with the two
rows above it. Its p99 *is* new information: a request can be bad at TTFT or at
TPOT without being bad at both, and only E2EL's tail shows how often they land
together.

There is no NTPOT. Dividing whole-request latency by token count gives a
stalled prefill and a uniformly slow decode the same value, which is what ITL
exists to distinguish. It used to be recorded and withheld; a number that must
never be read is not a measurement.

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
walnut bench serve --shape chat --num-prompts 120 --request-rate 16 \
  --goodput ttft:250 --goodput tpot:10 -o chat-rate16.json
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

                           mean      p50      p90      p99      max      std        n
TTFT (ms)                 26.33    23.13    40.59    49.05    53.08    7.340      120
TPOT (ms)                  4.39     4.59     5.63     6.22     6.25    1.008      120
ITL (ms)                   4.39     3.35     4.55    23.82    25.85    4.690    15240
E2EL (ms)                584.22   609.20   744.71   819.21   830.95  129.925      120
```

**Rate and concurrency are different knobs.** `--request-rate` is the traffic:
requests are submitted on a Poisson arrival process whether or not the server
keeps up. This is open-loop, and the only mode that builds a queue. Left off it
fires everything at once, which measures a saturated engine and says nothing
about queueing.

Poisson is fixed, not tunable. The general form is a gamma process with a shape
parameter for how much arrivals clump, and there is no traffic trace here to
calibrate that shape against — so any value but Poisson would be a number
picked to produce a result.

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
`itl`, `e2el`. An `itl` SLO is held against the request's **worst** gap.

**Every request is held to exactly `--max-tokens`**, and there is no way to
turn that off. A server that does not honour `ignore_eos` is a hard error:
ragged lengths mean the latencies cannot be compared with each other, and a
flag to produce them would only produce numbers this harness already refuses.

**`--profile`** wraps the measured window in `/start_profile` and
`/stop_profile` (needs `WALNUT_TORCH_PROFILER_DIR` on the server) and prints
the trace path. Profiled timings are inflated — take the trace from that run
and the numbers from a clean one.

## throughput

```bash
walnut bench throughput Qwen/Qwen3.5-0.8B --shape chat --num-prompts 200 \
  --max-batch-size 16
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

Two axes, and they answer different questions. Pass one.

Both are spelled exactly as `serve` spells them, and either may be given a
comma-separated ladder. **A comma is the whole signal**: `--request-rate 16`
means on `sweep` what it means on `serve`, and `--request-rate 8,16,24` sweeps.
Whichever flag carries the ladder is the axis; exactly one may.

So a single `--max-concurrency` beside a `--request-rate` ladder is a fixed
gate, exactly as it is on `serve`. One quantity, one name, one meaning
everywhere.

### a `--request-rate` ladder — open loop, find the knee

```bash
walnut bench sweep --shape chat --request-rate 8,16,24,32 --num-prompts 200 \
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
called for — deliberately not against `--request-rate`, since a finite Poisson
sample has a realized mean of its own.

### a `--max-concurrency` ladder — closed loop, draw the frontier

```bash
walnut bench sweep --shape chat --max-concurrency 1,2,4,8,16 --num-prompts 200 \
  --goodput tpot:10 -o frontier.json
```

The layout, with the columns filled in from the `micro` table further down so
the shape of the curve is real — the `p99` and goodput columns are sketched:

```
  conc  out tok/s  tok/s/user  TPOT p50  TPOT p99  ITL p99  E2EL p99  goodput
     1       602.5       602.5      1.50         .     1.61         .        .
     2       885.4       442.7      2.05         .     2.29         .        .
     4      1369.4       342.4      2.57         .    22.27         .        .
     8      1980.4       247.6      3.73         .    23.72         .        .
    16      2286.0       142.9      6.59         .    26.56         .        .

1980 tok/s at concurrency 8 — the most throughput on this ladder with every
request inside tpot p100 <= 10 ms.
```

**Every rung runs; there is no knee.** A closed loop holds a fixed number of
requests in flight, so nothing queues without bound and no rung invalidates the
ones above it. Each is a real operating point, and the curve is the answer.

**The last line is the number to quote.** System throughput at a stated
interactivity, in one figure, which cannot be repeated without its workload
attached. Without `--goodput` there is nothing to read it against and the table
has to be eyeballed.

**`tok/s/user` is throughput over mean concurrency** — the axis the public
benchmarks plot, printed so a walnut number can sit beside theirs. It is an
aggregate ratio, so it cannot tell a uniformly slow decode from one starved
stream; that is the same objection this harness makes to NTPOT. **Read
`TPOT p99` instead**, and use `tok/s/user` only to compare with somebody else's
chart.

**Which axis.** The rate ladder is the stronger question here and the only one
that sees queueing — it is what a serving change is judged on, and it is
what MLPerf's Server scenario reports (Poisson arrivals at a target QPS, the
result being the highest QPS still inside its TTFT and TPOT bounds).
The concurrency ladder is what the hardware-comparison reports use, and is for
sizing and for quoting: what does a batch of 8 buy, at what per-stream cost. Under offered traffic concurrency is an outcome and not a setting, so the
interactivity axis does not exist there; under a closed loop nothing ever
queues, so there is nothing to overload.

## latency

```bash
walnut bench latency Qwen/Qwen3.5-0.8B --shape chat --label baseline -o before.json
```

The shape's output length, 3 warm-up iterations, 5 measured, greedy, driven
straight at the model — no scheduler, no HTTP, detokenization after the clock stops. Greedy
is what makes the output hash a correctness check; it also never runs the
sampler's softmax path, so raise `--temperature` (with `--seed`) if the sampler
is what you changed.

**`--temperature` and `--top-p` live here and nowhere else.** They are the only
place a sampler change is legible: under `serve` the softmax and the nucleus
sort sit beneath queueing, batching and HTTP, and under `throughput` beneath a
whole batch. `serve` and `throughput` decode greedily, always.

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

The first request is always timed.

```
                         median     mean      std        n
load weights (s)          1.587    1.603    0.023        3
prepare batch (s)         0.144    0.144    0.000        3
first request (s)         0.031    0.031    0.000        3
total (s)                 1.763    1.778    0.023        3
```

Three phases, paid at different times and fixed by different work. Each
iteration builds a whole engine and throws it away, so the warm-ups absorb the
cold compile and what remains is what a restart pays.

A large `first request` means something `start` should have done up front is
being deferred into a request — which is why it cannot be switched off.

## Believing a delta

**Read the `std` column, and want ~2× it.** On a quiet RTX 5090 at default
iteration counts `latency` gives TTFT std ≈ 0.4 ms on a ~19 ms median and TPOT
std ≈ 0.011 ms on a ~1.6 ms median — so a TPOT change under ~0.02 ms, about 1%,
has not been shown. It converges by about ten iterations: raise `--num-iters`
for a tighter floor and it will not drift for having done so.

`serve`'s `std` is not a noise floor — its samples are 120 different requests.
There the run-to-run floor is about 2% on medians and p99s at 120 requests, and
under 0.1% on throughput. Fewer requests, wider tails.

Two runs are only comparable if they measured the same thing — same model,
device, dtype, prompt, token count and flags. Nothing enforces that; check it.

## What this machine does

RTX 5090, Qwen3.5-0.8B, measured before `--shape` existed, on what was then the
default: a seven-token prompt at 128 output tokens. That shape is gone, and it
was nearly all decode — so none of this describes what a 1k, 4k or 16k prefill
does to the same engine. Read the ratios, not the absolutes, and **re-measure
at `chat` before quoting anything.**

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

- **Decode kernels, fusion, graph contents** → `latency` TPOT, against its `std`.
  ITL p99 tracks the mean there, so read `ITL max` for genuine spikes.
- **Scheduler, batching, admission** → `serve` output tok/s *and* ITL p99. A
  change that raises throughput while widening the ITL tail traded away what
  users feel.
- **Prefill, chunking, prompt handling** → `serve` TTFT p50 and p99, at a
  finite `--request-rate` so prefills contend.
- **Batch sizing** → `throughput` across `--max-batch-size`.
- **Prefill cost, chunking, admission** → any `serve` metric across `--shape`.
  `chat` → `agentic` is 16× the prompt work at a quarter of the output.
- **Capacity, "how many can it take"** → a `sweep` rate ladder. The rung it
  stops on is the answer.
- **"How fast is it, in one number"** → throughput at an SLO.
- **Compilation, autotuning, lazy init** → `startup`.
- **TTFT alone never justifies a change.** `--no-cuda-graph` *improves* TTFT
  here by 3.7% while TPOT goes 1.48 → 3.37 ms and per-stream throughput falls
  56%. A run reporting only TTFT would have called that a win.

## Method

- **Baseline first**, before you edit.
- **Say which shape.** A tok/s without one describes an unknown workload.
- **Quiet machine, same machine.** GPU clocks drift with temperature, and the
  "after" run is always the one at the end of a long session.
- **Say what the workload was.** `tok/s` without the concurrency, shape and
  token counts is not a result. Every record carries all of them.
- **Match the profile.** `walnut profile` takes `--temperature` and
  `--max-tokens`; set them to the benchmark's, or the trace describes a
  different workload than the numbers beside it.
