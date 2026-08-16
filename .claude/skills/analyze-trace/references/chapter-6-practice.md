# Chapter 6 — Practice

> An end-to-end diagnosis, tail-latency work, regression testing, and the mistakes that invalidate results.

## 6.1 Worked case study: 485 → 2457 tok/s

The complete diagnostic path on the running example, from a trace to a fix, using nothing but the queries above.

### Step 0 — Preflight (§2.3)
```
stats: clean (no dropped slices)
categories: cpu_op 26338 / cuda_runtime 3902 / kernel 3572 / user_annotation 33
link coverage: 3604 device ops, 3604 linked
phases: decode x32 (avg 2055 µs), prefill x1 (4172 µs)
```
Trace is sound. Note the first ratio already: **26,338 ATen ops to produce 3,572 kernels.**

### Step 1 — Utilization (§3.3.1)
```
e2e 65.92 ms   gpu_busy 12.53 ms   utilization 19.0%
```
The GPU is idle 81% of the time. Whatever is slow, it is not the kernels. Everything about "which kernel should I optimize" is now off the table.

### Step 2 — Queue latency (§3.3.2)
```
decode: avg_queue 3.3 µs, 87% under 5 µs
```
Kernels start essentially the instant they are launched — the queue is empty. The GPU is starving, not backlogged. **Host-bound confirmed.**

### Step 3 — Host decomposition (§3.3.3, §3.2.1)
```
wall        2103 µs
├─ CUDA API  325 µs  (15%)  — 108 launches @ 2.7 µs, 1 sync @ 28 µs
├─ GPU busy  387 µs  (18%)  — overlapped with the above
└─ residual 1391 µs  (66%)  — Python + ATen dispatch
```
CUDA API is only 15%. This is **dispatch-bound**, not merely launch-bound — a distinction that matters, because pure kernel fusion would only address the 15%.

### Step 4 — Localize the host cost (§4.1)
```
798 ATen ops per token, max nesting depth 5, 7.4 ops per launch
top self-time: aten::mm 232 µs, aten::empty 175 µs (x100), aten::transpose 101 µs (x73),
               aten::as_strided 43 µs (x139)
```
A large fraction of host time is metadata ops (`as_strided`, `transpose`, `slice`) and allocator calls (`empty`) that produce no GPU work at all.

### Step 5 — Confirm with gap shape (§3.4.3)
```
3487 gaps totalling 53.4 ms; distribution concentrated at 10-20 µs
```
Uniform small gaps after every kernel — not a few discrete stalls. This rules out the sync as the primary cause (it is only 28 µs of 2103) and confirms per-kernel host overhead.

### Step 6 — Establish the ceiling (§3.2.2)
```
GPU-side floor: 391.6 µs/token → 2554 tok/s
current:       2053.2 µs/token →  487 tok/s
headroom: 5.2x
```

### Step 7 — Choose the fix
The diagnosis is: ~110 kernels per token, each 3.6 µs, each preceded by ~7 ATen dispatches and a 2.7 µs launch, with the GPU idle throughout. The intervention that removes *all* per-token host work at once is CUDA graph capture of the decode step. (`torch.compile(mode="reduce-overhead")` is the same fix packaged; it adds fusion, which would additionally attack the 22% of device time spent in 1,824 tiny norm/elementwise kernels.)

### Step 8 — Verify (§4.3.1, §4.3.3)
```sql
-- capture verified:
cudaGraphLaunch  x32 @ 46.9 µs      graph id=2: 3424 kernels      graph id=0: 147 (prefill+sampling)
```
```
                   before      after     delta
decode e2e        65.92 ms   13.02 ms    5.06x
throughput       485 tok/s  2457 tok/s   5.06x
mean ITL          2053 µs      74 µs    27.7x
GPU busy          12.53 ms   12.51 ms    1.00x
GPU utilization     19.0 %     96.1 %    5.06x
ATen ops (trace)     26338        927    28.4x
host launch cost   9.34 ms    1.75 ms    5.34x
```

Achieved 2457 tok/s against a predicted ceiling of 2554 — within 4%. The model is now GPU-bound at 96% utilization (`queue_ns` flipped from 3.3 µs to 5.2 ms), and the next lever is a device-side one: fuse the norms and elementwise ops, or quantize the weights to cut the gemv bandwidth that is 60% of remaining device time.

**The general lesson.** Six queries, none of which looked at a kernel, produced the diagnosis. The kernel inventory in §5.1 was only useful *after* the host problem was fixed — and it immediately named the next target. Order matters: **utilization → queue → host split → localize → fix → re-measure.**

## 6.2 Tail latency and outlier tokens

Serving SLOs are p99, not mean. The trace lets you find and explain the tail.

```sql
WITH d AS (SELECT seq, dur FROM phase WHERE name='decode' AND seq > 0)
SELECT COUNT(*) n,
       AVG(dur)/1e3 AS mean_us,
       MIN(dur)/1e3 AS min_us,
       MAX(dur)/1e3 AS max_us,
       (SELECT dur FROM d ORDER BY dur LIMIT 1 OFFSET (SELECT CAST(COUNT(*)*0.50 AS INT) FROM d))/1e3 AS p50_us,
       (SELECT dur FROM d ORDER BY dur LIMIT 1 OFFSET (SELECT CAST(COUNT(*)*0.95 AS INT) FROM d))/1e3 AS p95_us,
       (SELECT dur FROM d ORDER BY dur LIMIT 1 OFFSET (SELECT CAST(COUNT(*)*0.99 AS INT) FROM d))/1e3 AS p99_us
FROM d;
```

Then explain the worst tokens by diffing them against the median:

```sql
WITH d AS (SELECT seq, ts, te, dur FROM phase WHERE name='decode' AND seq > 0),
     med AS (SELECT dur FROM d ORDER BY dur LIMIT 1 OFFSET (SELECT COUNT(*)/2 FROM d))
SELECT d.seq, d.dur/1e3 AS wall_us,
       (SELECT SUM(gdur)/1e3 FROM link l WHERE l.ph='decode' AND l.seq=d.seq)     AS gpu_us,
       (SELECT SUM(dur)/1e3  FROM api a WHERE a.ts>=d.ts AND a.ts<d.te)           AS api_us,
       (SELECT COUNT(*)      FROM ev e  WHERE e.cat='cpu_op' AND e.ts>=d.ts AND e.ts<d.te) AS n_ops,
       (SELECT e.name FROM ev e WHERE e.ts>=d.ts AND e.ts<d.te AND e.cat='cpu_op'
         ORDER BY e.dur DESC LIMIT 1)                                             AS longest_op
FROM d WHERE d.dur > (SELECT dur FROM med) * 1.5
ORDER BY d.dur DESC LIMIT 10;
```

Common causes, and their signatures:

| Cause | Signature |
|---|---|
| Allocator growth (`cudaMalloc`) | a `cudaMalloc` slice in `api`; `Total Reserved` steps up in the same window |
| Attention kernel switching at a sequence-length boundary | `gpu_us` jumps; `kname` differs from the median token's |
| Scheduler / batch composition change | `n_ops` differs; extra annotations present |
| Recompilation | multi-ms `cpu_op`, no CUDA activity |
| Another process on the GPU | `gpu_us` per kernel inflates uniformly with no host change |

The fourth column, `n_ops`, is the fastest discriminator: if the outlier token executed the *same* number of ATen ops, the host did the same work and the problem is on the device or in the allocator; if it executed more, the problem is in your control flow.

## 6.3 A/B trace comparison

Every optimization needs a before/after. Load both traces and compare; the shape below generalizes to any metric in this book.

```python
from perfetto.trace_processor import TraceProcessor
import pandas as pd

PRELUDE = open('prelude.sql').read()

METRICS = {
  'gpu_busy_ms':   "SELECT SUM(gdur)/1e6 v FROM link WHERE ph='decode'",
  'host_api_ms':   "SELECT SUM(ldur/launch_fanout)/1e6 v FROM link WHERE ph='decode'",
  'n_kernels':     "SELECT COUNT(*) v FROM link WHERE ph='decode'",
  'n_aten_ops':    "SELECT COUNT(*) v FROM ev WHERE cat='cpu_op'",
  'mean_itl_us':   "SELECT AVG(dur)/1e3 v FROM phase WHERE name='decode' AND seq>0",
  'gpu_util_pct':  """WITH d AS (SELECT MIN(ts) a FROM phase WHERE name='decode'),
                           e AS (SELECT MAX(te) b FROM dev_op)
                      SELECT (SELECT SUM(dur) FROM dev_op WHERE ts>=(SELECT a FROM d))*100.0
                             /((SELECT b FROM e)-(SELECT a FROM d)) v""",
  'sync_us':       "SELECT SUM(dur)/1e3 v FROM api WHERE name GLOB '*Synchronize*'",
}

def measure(path):
    tp = TraceProcessor(trace=path)
    for stmt in PRELUDE.split(';'):
        if stmt.strip(): tp.query(stmt)
    return {k: next(iter(tp.query(q))).v for k, q in METRICS.items()}

a, b = measure('before.json'), measure('after.json')
df = pd.DataFrame({'before': a, 'after': b})
df['ratio'] = df['before'] / df['after']
print(df.round(3))
```

Per-kernel regression hunting — which specific kernels changed?

```python
KQ = """SELECT kname, COUNT(*) n, SUM(gdur)/1e3 us FROM link WHERE ph='decode' GROUP BY 1"""
ka = measure_df('before.json', KQ).set_index('kname')
kb = measure_df('after.json',  KQ).set_index('kname')
delta = kb.join(ka, rsuffix='_before', how='outer').fillna(0)
delta['d_us'] = delta.us - delta.us_before
print(delta.sort_values('d_us').head(15))   # biggest wins
print(delta.sort_values('d_us').tail(15))   # biggest regressions
```

**Comparison hygiene.** Capture both traces on the same machine, same driver, same clocks (`nvidia-smi -lgc` to lock if you can), with identical warm-up and identical token counts, and always drop `seq = 0`. Compare *distributions*, not single tokens. A 5% delta in mean ITL from a single 32-token capture is noise.

## 6.4 Automation and CI

### 6.4.1 Ship the prelude as a SQL package

```bash
mkdir -p sqlpkg/torchinf && cp prelude.sql sqlpkg/torchinf/prelude.sql
tp query --add-sql-package ./sqlpkg -f check.sql trace.json
```
with `check.sql` beginning `INCLUDE PERFETTO MODULE torchinf.prelude;`. This keeps one authoritative copy of the view layer across scripts, notebooks, and CI.

### 6.4.2 A regression gate

```python
THRESHOLDS = {
    'gpu_util_pct': ('min', 80.0),     # decode must keep the GPU busy
    'n_aten_ops':   ('max', 2000),     # graphs/compile must stay engaged
    'sync_us':      ('max', 200.0),    # no new .item() in the hot loop
    'mean_itl_us':  ('max', 90.0),     # the SLO itself
}

m, failures = measure('trace.json'), []
for k, (kind, bound) in THRESHOLDS.items():
    v = m[k]
    if (kind == 'min' and v < bound) or (kind == 'max' and v > bound):
        failures.append(f"{k}={v:.2f} violates {kind} {bound}")
if failures:
    raise SystemExit("PERF REGRESSION:\n" + "\n".join(failures))
```

`n_aten_ops` is the highest-value guard in practice: a single graph break or an accidental `.cpu()` moves it by an order of magnitude and it has no measurement noise. `gpu_util_pct` is the best single summary. Both are far more stable in CI than wall-clock timing.

### 6.4.3 Continuous capture in production

Capture a short window periodically (`schedule(wait=N, warmup=1, active=2, repeat=1)` around the serving loop), gzip, upload, and run the gate offline. Trace size is the constraint: budget ~15 MB per 3,500 kernels uncompressed, ~1.5 MB gzipped. Keep `with_stack=False` in production for both size and the slice-drop bug.

## 6.5 Twelve traps

1. **Forgetting the `args.` prefix.** `extract_arg(id, 'correlation')` returns NULL silently. It is `args.correlation`.
2. **Assuming microseconds.** JSON is µs; the `slice` table is ns.
3. **Profiling cold.** Token 0 in our trace was 70% slower than steady state. Always `WHERE seq > 0`.
4. **Stopping the profiler with work in flight.** Missing `torch.cuda.synchronize()` truncates the GPU timeline and fabricates a host-bound diagnosis.
5. **`with_stack=True` silently dropping your annotations.** `slice_drop_overlapping_complete_event`. Check `stats` every time.
6. **Attributing phases by slice ancestry.** Off-main-thread work has a different depth-0 root. Attribute by launch timestamp.
7. **Ignoring `cuda_driver`.** cuBLASLt and CUTLASS use `cuLaunchKernel`, not `cudaLaunchKernel`. Filtering only `cuda_runtime` lost 144 of 899 launches in our training trace — 16%.
8. **Summing launch duration under CUDA graphs.** 160 ms instead of 1.75 ms. Divide by `launch_fanout`.
9. **Treating `correlation` as a key.** 3,571 kernels shared 179 correlation ids under graph replay. Use `flow`.
10. **Summing kernel durations across streams to get "GPU time".** Overlapping kernels double-count. Compute the interval union (§5.4.2).
11. **Measuring a phase window to the last annotation rather than the last device op.** Under graphs the host finishes 5× early; the window collapses and utilization reads as 93% when the honest number is 96% measured over a 5× longer span — and in other traces the error goes the other way and is much larger.
12. **Trusting `est. achieved occupancy %` as a measurement.** It is derived from launch geometry, is NULL/0 for some kernels, and does not predict throughput.


---

[Index](README.md) · [← Chapter 5](chapter-5-device-side-bottlenecks.md)
