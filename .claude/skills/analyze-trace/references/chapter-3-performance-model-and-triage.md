# Chapter 3 — The Performance Model and Bottleneck Triage

> The prefill/decode model, per-token accounting, and a graph check plus three-test procedure that identifies which bottleneck you have before you try to fix one.

## 3.1 The inference performance model

Inference has two regimes with opposite bottlenecks. Every analysis begins by establishing which one you are in.

**Prefill** processes `S` prompt tokens at once. GEMMs are `[B·S, d] × [d, k]` — compute-bound, high arithmetic intensity, tensor cores engaged. Kernel durations are large enough that host overhead is amortized. Prefill determines **TTFT** (time to first token).

**Decode** processes one token per sequence. GEMMs collapse to `[B, d] × [d, k]` — matrix-*vector* products at `B=1`, memory-bandwidth-bound, arithmetic intensity ≈ 1. Kernel durations collapse with them, and host overhead does not. Decode determines **ITL** (inter-token latency) and throughput.

The regime change is visible directly in the trace:

```sql
SELECT ph, COUNT(*) n, SUM(gdur)/1e6 ms, AVG(gdur)/1e3 avg_us,
       SUM(CASE WHEN gdur < 5000 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_under_5us
FROM link GROUP BY 1;
```
```
ph        n      ms       avg_us   pct_under_5us
decode    3488   12.531   3.593    68.8
prefill    116    0.998   8.602    63.8
```

Decode issues **30× more kernels** than prefill to do a fraction of the work, and 69% of them run for under 5 µs. A kernel that runs for 3.6 µs cannot amortize a 2.7 µs launch, let alone the ~15 µs of host work that produced the launch. **This is the structural reason decode is host-bound in eager PyTorch, and it is why CUDA graphs (§4.3) are not an optimization but a requirement.**

### 3.1.1 The bottleneck classes

| Class | Signature in the trace | Fix |
|---|---|---|
| **Host-bound (dispatch)** | GPU idle ≫ 0; `cpu_op` self-time ≫ CUDA API time; hundreds of aten ops per token | CUDA graphs, `torch.compile`, fewer/fused ops, C++ runtime |
| **Launch-bound** | GPU idle ≫ 0; CUDA API time dominates host time; `queue_ns` ≈ 0 | CUDA graphs, kernel fusion, larger batch |
| **GPU-bound** | GPU idle ≈ 0; `queue_ns` large and growing | Better kernels, quantization, more/larger batch, better parallelism |
| **Sync-bound** | Large `cudaStreamSynchronize`/`cudaMemcpyAsync DtoH` slices at a fixed point per token | Defer `.item()`, sample on device, async stopping criteria |
| **Transfer-bound** | `gpu_memcpy` a significant share of device time; low `memory bandwidth (GB/s)` | Pin memory, use `non_blocking=True`, keep tensors resident, overlap on a copy stream |
| **Mixed** | GPU idle is real but not dominant (utilization 60–85%); no single class above accounts for most of the token | Attribute the idle (§3.4.2), then fix whichever of the above the attribution names — usually device *and* host in sequence |

## 3.2 Phase segmentation and the per-token budget

### 3.2.1 The master query

This one table answers "where does a token go" and is the starting point for every investigation.

```sql
SELECT p.seq AS tok,
       p.dur/1e3                                                          AS wall_us,
       (SELECT SUM(gdur)/1e3 FROM link l
         WHERE l.ph = 'decode' AND l.seq = p.seq)                         AS gpu_busy_us,
       (SELECT COUNT(*)      FROM link l
         WHERE l.ph = 'decode' AND l.seq = p.seq)                         AS n_kernels,
       (SELECT SUM(dur)/1e3  FROM api a
         WHERE a.ts >= p.ts AND a.ts < p.te)                              AS cuda_api_us,
       (SELECT SUM(dur)/1e3  FROM api a
         WHERE a.ts >= p.ts AND a.ts < p.te AND a.name GLOB '*Synchronize*') AS sync_us,
       (SELECT COUNT(*)      FROM ev e
         WHERE e.cat = 'cpu_op' AND e.ts >= p.ts AND e.ts < p.te)         AS n_aten_ops
FROM phase p
WHERE p.name = 'decode'
ORDER BY tok;
```
```
tok  wall_us   gpu_busy_us  n_kernels  cuda_api_us  sync_us  n_aten_ops
0    2114.7    392.9        109        342.3        28.6     798
1    2102.8    386.9        109        325.1        28.2     798
2    2032.7    396.5        109        325.4        28.0     798
3    1998.5    388.3        109        328.0        28.7     798
4    2060.9    393.9        109        327.9        32.8     798
```

Read the first row of the steady state (`tok = 1`) as an accounting identity:

```
wall            2103 µs      100%
├─ GPU busy      387 µs       18%    the only useful work
├─ CUDA API      325 µs       15%    108 launches + 1 sync
│   └─ sync       28 µs        1%    the .item() stall
└─ unaccounted  1391 µs       66%    ← Python + ATen dispatch
```

Two thirds of every token is spent in the PyTorch dispatcher, executing 798 ATen operators to produce 109 kernels. The GPU is idle for 82% of the token. No kernel-level optimization can help this workload; the diagnosis is *host-bound*, and §4.1 localizes it further.

### 3.2.2 Aggregate throughput, dropping warm-up

```sql
SELECT COUNT(*) tokens,
       AVG(dur)/1e3 AS avg_itl_us,
       1e9/AVG(dur) AS tok_per_s,
       (SELECT AVG(x) FROM (SELECT SUM(gdur) x FROM link
                            WHERE ph='decode' AND seq>0 GROUP BY seq))/1e3 AS avg_gpu_us,
       (SELECT AVG(x) FROM (SELECT COUNT(*)  x FROM link
                            WHERE ph='decode' AND seq>0 GROUP BY seq))     AS avg_kernels
FROM phase WHERE name='decode' AND seq > 0;
```
```
tokens=31  avg_itl_us=2053.2  tok_per_s=487.0  avg_gpu_us=391.6  avg_kernels=109.0
```

Ceiling estimate: if the host cost went to zero, `1e6/391.6 = 2554 tok/s`. That is a 5.2× headroom figure and it is what you quote when arguing for the fix.

### 3.2.3 TTFT

```sql
SELECT ((SELECT ts + dur FROM phase WHERE name='decode' ORDER BY ts LIMIT 1)
      - (SELECT ts       FROM phase WHERE name='prefill'))/1e6 AS ttft_ms;
```
```
ttft_ms = 6.296     -- prefill 4.17 ms + first decode 2.11 ms
```

Decompose it the same way as a decode token, but expect the opposite verdict: prefill's 116 kernels averaged 8.6 µs and its GEMMs are `cutlass_80_tensorop_f16_s16816gemm` — tensor-core kernels doing real work.

### 3.2.4 The GPU-side phase span

Kineto emits a **`gpu_user_annotation`** slice mirroring each `user_annotation`, placed on the device track and spanning the device ops attributable to it. This gives you the GPU-side view of a phase for free — no joins, no correlation, and it survives the `with_stack` slice-drop bug.

```sql
SELECT name, COUNT(*) n, AVG(dur)/1e3 avg_us, SUM(dur)/1e6 ms
FROM slice WHERE category = 'gpu_user_annotation' GROUP BY 1;
```
```
eager:      decode  32  avg 2019.4 µs      prefill  1  avg 2375.6 µs
cudagraph:  decode  32  avg  401.4 µs      prefill  1  avg 1959.8 µs
```

In eager mode the GPU-side decode span (2019 µs) nearly equals the host-side span (2055 µs) — the GPU is being dragged along in lockstep by the host, idling between kernels. Under CUDA graphs it collapses to 401 µs, close to the 391 µs of actual kernel time: the GPU is now nearly saturated within each token. **The ratio `gpu_busy / gpu_user_annotation.dur` is a one-query utilization metric.**

## 3.3 Bottleneck triage: the decision procedure

Establish graph state (§3.3.0), then run the three measurements in order. They
partition the space.

### 3.3.0 Before the tests — is a graph replaying?

CUDA graphs change what all three tests mean, so answer this first.

```sql
SELECT ph, AVG(launch_fanout) AS avg_fanout, MAX(launch_fanout) AS graph_size,
       COUNT(DISTINCT CASE WHEN graph_id > 0 THEN graph_id END) AS graphs
FROM link WHERE ph IS NOT NULL GROUP BY 1;
```

`graph_id` is `0`, not NULL, for kernels outside any graph (§4.3.1) — hence the
`CASE`, without which every eager phase reports one phantom graph. `graphs = 0`
and `avg_fanout ≈ 1` is eager: one launch, one kernel. `avg_fanout ≫ 1` means a
single `cudaGraphLaunch` is submitting that many kernels. The average is
*kernel*-weighted, so a phase that mixes graph and eager launches reports less
than the true graph size; `graph_size` is that number.

Three things follow, one per test:

- **Test 1.** Utilization is usually high *within* a replay, so a mediocre
  number points at the gaps *between* replays — submission cost and syncs — not
  at the kernels. Go to §3.4.2.
- **Test 2.** Per-kernel `queue_ns` is no longer meaningful — see the caveat in
  §3.3.2 and read it at replay granularity.
- **Test 3.** Its total is measured over `api`, one row per call, so the number
  stays correct — but "API ≫ rest" no longer means launch-bound in the eager
  sense, because a handful of `cudaGraphLaunch` calls now dominate it. Read it
  with §4.3.2. (The `launch_fanout` division in §6.5 trap 8 applies to
  aggregates over `link`, where `ldur` repeats once per kernel — *not* to this
  query.)

The same reasoning applies to `torch.compile(mode="reduce-overhead")`, which
captures graphs underneath.

### 3.3.1 Test 1 — GPU utilization over the decode window

```sql
WITH d AS (SELECT MIN(ts) a, MAX(ts+dur) b FROM phase WHERE name='decode'),
     k AS (SELECT ts, ts+dur te FROM dev_op WHERE ts >= (SELECT a FROM d)),
     e AS (SELECT MAX(te) e FROM k)
SELECT ((SELECT e FROM e) - (SELECT a FROM d))/1e6                          AS e2e_ms,
       (SELECT SUM(dur) FROM dev_op WHERE ts >= (SELECT a FROM d))/1e6      AS gpu_busy_ms,
       (SELECT SUM(dur) FROM dev_op WHERE ts >= (SELECT a FROM d)) * 100.0
         / ((SELECT e FROM e) - (SELECT a FROM d))                          AS gpu_util_pct;
```
```
eager:      e2e_ms=65.92   gpu_busy_ms=12.531   gpu_util_pct=19.0
cudagraph:  e2e_ms=13.02   gpu_busy_ms=12.512   gpu_util_pct=96.1
```

Note the window is measured to the **last device op**, not the last annotation — under CUDA graphs the host finishes issuing long before the GPU finishes executing, and measuring to the annotation would report a nonsensical 2.5 ms.

- **Utilization > 85%** → GPU-bound. Go to §5.1, §5.2, §5.5.
- **Utilization 60–85%** → **mixed**. Typical of a stack that has already had
  its worst host problem fixed — the two traces above sit at 19% and 96%
  precisely because each has one dominant cause, and a partly optimized one
  lands between. It is not a tie-breaker and does not mean "nearly GPU-bound":
  run tests 2 and 3 anyway, then attribute the idle with §3.4.2. Report the
  device cost and the host cost side by side; both are usually worth fixing,
  and the gap attribution tells you which comes first.
- **Utilization < 60%** → the GPU is starving. Continue to test 2.

### 3.3.2 Test 2 — queue latency

`queue_ns = kernel.ts − launch_call.end_ts`: how long a kernel waited between its launch returning and its execution starting. This is the sharpest single discriminator in GPU performance analysis.

```sql
SELECT ph, COUNT(*) n,
       AVG(queue_ns)/1e3                                              AS avg_queue_us,
       SUM(CASE WHEN queue_ns < 2000 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_immediate,
       SUM(CASE WHEN queue_ns > 50000 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_deep_backlog
FROM link GROUP BY 1;
```

Interpretation:

| `queue_ns` | Meaning | Verdict |
|---|---|---|
| ≈ 0 (immediate) | The GPU was idle, waiting; execution began the instant work arrived | **Host/launch-bound** |
| Large and stable | A steady backlog: the host is ahead, the GPU is the constraint | **GPU-bound** |
| Large and *growing* within a phase | The host is issuing faster than the GPU drains | GPU-bound, and the host has headroom |
| Negative | Kernel started before the launch call returned — normal for long-running launch APIs; treat as immediate | Host-bound |

Our traces: eager decode `avg_queue = 3.3 µs` with 87% under 5 µs — the queue is empty, the GPU is waiting. Graph decode `avg_queue = 5.2 ms` — a deep, healthy backlog.

> **Caveat for graphs.** Inside a captured graph, a kernel's wait includes the execution of all preceding kernels in the same graph, since they share one launch. Per-kernel `queue_ns` is only meaningful for individually launched kernels; for graphs, interpret it at the *replay* granularity (first kernel of each replay).

### 3.3.3 Test 3 — host time decomposition

If test 1 came in under 85% — host-bound or mixed — split the host time:

```sql
WITH p AS (SELECT ts, te, dur FROM phase WHERE name='decode' AND seq BETWEEN 1 AND 10)
SELECT SUM(p.dur)/1e3                                                        AS wall_us,
       (SELECT SUM(a.dur) FROM api a, p WHERE a.ts>=p.ts AND a.ts<p.te)/1e3  AS cuda_api_us,
       (SELECT SUM(a.dur) FROM api a, p
         WHERE a.ts>=p.ts AND a.ts<p.te AND a.name GLOB '*Synchronize*')/1e3 AS sync_us
FROM p;
```

- **CUDA API ≫ the rest** → *launch-bound*. Reduce the number of launches: fusion, graphs.
- **CUDA API is a small fraction** (our case: 325 of 2103 µs) → *dispatch-bound*. The time is in Python and the ATen dispatcher. §4.1.
- **Sync dominates** → *sync-bound*. §4.2.

### 3.3.4 Triage summary

```
first: avg_fanout > 1 ? ───► graphs are replaying; reinterpret all three tests (§3.3.0)

util > 85% ────────────────────────────────► GPU-bound      → §5.1 §5.2 §5.5
util 60-85% ───────────────────────────────► mixed; run tests 2+3, then §3.4.2
util < 60% ─┬─ queue ≈ 0 ─┬─ API ≫ rest ───► launch-bound   → §4.3 §4.4
            │             ├─ API ≪ rest ───► dispatch-bound → §4.1 §4.3
            │             └─ sync dominates► sync-bound     → §4.2
            └─ queue large ─────────────────► GPU-bound with a stalled window → §3.4
```

## 3.4 Gap analysis with blame attribution

Utilization tells you *how much* the GPU idled. Gap analysis tells you *when* and *because of what*.

### 3.4.1 Computing gaps

Device ops on one stream are ordered and non-overlapping, but across streams they are not. The general form uses a running maximum of end times to build the union of busy intervals; gaps are where the next start exceeds it.

```sql
WITH g AS (SELECT gts, gte FROM link WHERE ph = 'decode'),
     m AS (SELECT gts, gte,
                  MAX(gte) OVER (ORDER BY gts
                                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_max
           FROM g)
SELECT COUNT(*)                     AS n_gaps,
       SUM(gts - prev_max)/1e6      AS idle_ms,
       MAX(gts - prev_max)/1e3      AS max_gap_us,
       AVG(gts - prev_max)/1e3      AS avg_gap_us
FROM m WHERE prev_max IS NOT NULL AND gts > prev_max;
```

Eager decode: **3,487 gaps totalling 53.4 ms** against 12.5 ms of work. The gaps are not a few big stalls — they are ~15 µs of dead air after *every single kernel*. That distribution is the fingerprint of dispatch-bound decode, and it tells you immediately that hunting for one bad sync would be a waste of time.

### 3.4.2 Blaming a gap

For the traces where gaps *are* few and large, attribute each one to the host activity that spans it:

```sql
WITH g AS (SELECT gts, gte FROM link WHERE ph='decode'),
     m AS (SELECT gts, MAX(gte) OVER (ORDER BY gts
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS pm FROM g),
     gaps AS (SELECT pm AS gap_start, gts AS gap_end, gts - pm AS gap
              FROM m WHERE pm IS NOT NULL AND gts > pm)
SELECT gap/1e3 AS gap_us,
       -- longest CUDA API call overlapping the gap
       (SELECT e.name FROM ev e
         WHERE e.cat IN ('cuda_runtime','cuda_driver')
           AND e.ts < gaps.gap_end AND e.te > gaps.gap_start
         ORDER BY e.dur DESC LIMIT 1) AS blocking_call,
       -- innermost ATen op fully containing the gap
       (SELECT e.name FROM ev e
         WHERE e.cat = 'cpu_op' AND e.ts <= gaps.gap_start AND e.te >= gaps.gap_end
         ORDER BY e.depth DESC LIMIT 1) AS cpu_context
FROM gaps ORDER BY gap DESC LIMIT 10;
```

Sample output (from the training trace, where gaps are structured):
```
gap_us   blocking_call            cpu_context
42.72    cudaStreamSynchronize    (null)
31.97    cudaLaunchKernel         (null)
17.82    cudaMemsetAsync          aten::mm
 5.57    cudaStreamSynchronize    aten::copy_
 4.16    cuLaunchKernel           aten::matmul
```

Reading the two columns together:
- `blocking_call = *Synchronize*` → the host deliberately waited. Sync-bound; find out who asked (§4.2).
- `blocking_call = *Launch*` and `cpu_context` is an aten op → the host was simply slow to produce the next launch. Dispatch-bound.
- Both NULL → the host was outside CUDA entirely: Python, tokenizer, scheduler, logging, or the network. Enable `with_stack` (carefully) or annotate more finely.

### 3.4.3 Gap histogram

Shape matters more than the total:

```sql
WITH g AS (SELECT gts, gte FROM link WHERE ph='decode'),
     m AS (SELECT gts, MAX(gte) OVER (ORDER BY gts
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) pm FROM g),
     gaps AS (SELECT gts - pm AS gap FROM m WHERE pm IS NOT NULL AND gts > pm)
SELECT CASE WHEN gap <    1000 THEN 'a <1us'
            WHEN gap <    5000 THEN 'b 1-5us'
            WHEN gap <   20000 THEN 'c 5-20us'
            WHEN gap <  100000 THEN 'd 20-100us'
            WHEN gap < 1000000 THEN 'e 0.1-1ms'
            ELSE                    'f >1ms' END AS bucket,
       COUNT(*) n, SUM(gap)/1e6 ms
FROM gaps GROUP BY 1 ORDER BY 1;
```

- Mass in `<20 µs`, thousands of them → per-kernel host overhead → **graphs/fusion**.
- A handful in `>1 ms` → discrete stalls → **find them individually** (§3.4.2).
- Mass in `0.1–1 ms` → scheduler or batching stalls in a serving loop → instrument the scheduler with `record_function`.


---

[Index](README.md) · [← Chapter 2](chapter-2-capture-and-view-layer.md) · [Chapter 4 →](chapter-4-host-side-bottlenecks.md)
