# Chapter 4 — Host-Side Bottlenecks

> Dispatch overhead, synchronization stalls, and the two interventions that remove them: CUDA graphs and torch.compile.

## 4.1 Host-side cost: dispatch and Python

When §3.3.3 says dispatch-bound, this section finds the ops responsible.

### 4.1.1 Self time

Inclusive duration is useless for nested slices (`aten::matmul` contains `aten::mm`). Self time subtracts children — and because `cuda_runtime` slices are children of the op that launched them, self time correctly excludes launch cost too.

```sql
WITH p AS (SELECT ts, te FROM phase WHERE name='decode' AND seq=1)
SELECT s.name, COUNT(*) n,
       SUM(s.dur)/1e3 AS incl_us,
       SUM(s.dur - COALESCE((SELECT SUM(c.dur) FROM slice c WHERE c.parent_id = s.id), 0))/1e3
         AS self_us
FROM slice s, p
WHERE s.category='cpu_op' AND s.ts >= p.ts AND s.ts < p.te
GROUP BY 1 ORDER BY self_us DESC LIMIT 12;
```
```
name                                    n     incl_us   self_us
aten::mm                                33    324.1     232.0
aten::empty                            100    175.0     175.0
aten::transpose                         73    124.9     101.2
aten::scaled_dot_product_attention       8    449.0      99.4
aten::native_layer_norm                 17    219.1      85.1
aten::matmul                            33    420.5      67.5
aten::_flash_attention_forward           8    239.7      65.9
aten::add                               16    100.9      60.9
aten::copy_                             16     90.8      49.0
aten::as_strided                       139     43.4      43.4
aten::slice                             33     47.9      36.8
```

Read this as a dispatch census, not a compute census. `aten::empty` called 100 times for 175 µs is pure allocator overhead. `aten::as_strided` × 139 and `aten::transpose` × 73 are *metadata-only* operations — zero GPU work, ~0.6 µs of C++ dispatch each. There are **798 ATen ops per token**, nested up to depth 5, producing 109 kernels: a 7:1 ratio of dispatcher calls to actual GPU work.

### 4.1.2 The dispatch-overhead ratio

A single number to track:

```sql
SELECT (SELECT COUNT(*) FROM ev  WHERE cat='cpu_op' AND ts>=p.ts AND ts<p.te) AS aten_ops,
       (SELECT COUNT(*) FROM api WHERE ts>=p.ts AND ts<p.te
          AND name GLOB '*Launch*')                                          AS launches,
       (SELECT COUNT(*) FROM ev  WHERE cat='cpu_op' AND ts>=p.ts AND ts<p.te) * 1.0
     / (SELECT COUNT(*) FROM api WHERE ts>=p.ts AND ts<p.te AND name GLOB '*Launch*')
                                                                             AS ops_per_launch
FROM phase p WHERE name='decode' AND seq=1;
```
```
aten_ops=798   launches=108   ops_per_launch=7.4
```

Above ~3, the eager dispatcher is your bottleneck and `torch.compile` or CUDA graphs will pay for themselves. Below ~1.5, you are already close to the launch floor and further gains require fusion.

### 4.1.3 Python attribution (`with_stack=True`)

`python_function` slices carry names of the form `file.py(LINE): func`. Self time over them localizes host cost to source lines:

```sql
SELECT name, COUNT(*) n,
       SUM(dur - COALESCE((SELECT SUM(c.dur) FROM slice c WHERE c.parent_id = s.id),0))/1e3
         AS self_us
FROM slice s WHERE category='python_function'
GROUP BY 1 ORDER BY self_us DESC LIMIT 20;
```

Expect to see framework tax at the top — in our capture, `torch/_jit_internal.py(104): is_scripting` fired 665 times and `torch/compiler/__init__.py(477): is_compiling` 309 times *per capture*. These are the guards that `torch.compile` removes.

Remember §2.1.3: verify `stats` is clean, and if your annotations were dropped, fall back to `gpu_user_annotation` for phase windows.

## 4.2 The synchronization audit

Every host↔device sync is a pipeline bubble. In decode, one sync per token is enough to serialize everything.

```sql
SELECT a.name, COUNT(*) n, SUM(a.dur)/1e3 us, AVG(a.dur)/1e3 avg_us,
       (SELECT x.name FROM ancestor_slice(a.id) x ORDER BY x.depth DESC LIMIT 1) AS op,
       (SELECT x.name FROM ancestor_slice(a.id) x WHERE x.depth = 0)             AS root
FROM api a
WHERE a.name GLOB '*Synchronize*' OR a.name GLOB '*Memcpy*' OR a.name GLOB '*EventQuery*'
GROUP BY 1, 5 ORDER BY us DESC;
```
```
name                   n    us      avg_us  op
cudaStreamSynchronize  32   911.5   28.5    aten::_local_scalar_dense
cudaMemcpyAsync        32   171.6    5.4    aten::_local_scalar_dense
cudaDeviceSynchronize   2     4.5    2.2    (null)
```

`aten::_local_scalar_dense` is the ATen implementation of `Tensor.item()`. The trace has named the culprit exactly: 32 calls, one per token, 28.5 µs of blocking each — our `tok.item()` in the sampling loop. The `cudaMemcpyAsync` beside it is the 4-byte D2H transfer.

**The sync catalogue for inference:**

| Source op in the trace | Python cause | Fix |
|---|---|---|
| `aten::_local_scalar_dense` | `.item()`, `int(t)`, `float(t)` | Keep the token on device; batch the D2H |
| `aten::copy_` with `Memcpy DtoH` | `.cpu()`, `.numpy()`, `.tolist()` | Defer; transfer once per N tokens |
| `aten::nonzero`, `aten::unique`, `aten::masked_select` | data-dependent shapes | Restructure to fixed shapes |
| `aten::equal`, `aten::allclose` | stop-condition checks | Compute on device, sync every K steps |
| `cudaDeviceSynchronize` | `torch.cuda.synchronize()` | Remove from the hot loop |
| `cudaStreamSynchronize` with no op ancestor | allocator, or a framework `.to()` | Check `non_blocking=True` and pinning |

**Quantifying the cost.** The blocking *call* is 28.5 µs, but the true cost is the pipeline bubble it creates: the host cannot issue the next token's work until the GPU has drained. Measure it as the gap that follows:

```sql
WITH s AS (SELECT id, ts, te FROM api WHERE name GLOB '*Synchronize*')
SELECT s.id, (s.te - s.ts)/1e3 AS blocked_us,
       ((SELECT MIN(gts) FROM link WHERE gts > s.te) - s.te)/1e3 AS bubble_after_us
FROM s ORDER BY blocked_us DESC LIMIT 10;
```

A stop-string check that syncs every token can cost more than the model. The generic fix is to make the criterion asynchronous: evaluate on device into a flag tensor, and read the flag once every K tokens, accepting K−1 tokens of overrun.

## 4.3 CUDA graphs

For a launch- or dispatch-bound decode, CUDA graphs are the primary fix: capture the per-token kernel sequence once and replay it with a single host call, eliminating both dispatch and per-kernel launch cost.

### 4.3.1 Verifying that capture actually happened

Two independent checks, both mandatory — it is easy to believe you are using graphs when you are not.

```sql
-- 1. Are the launches graph launches?
SELECT name, COUNT(*) n, SUM(dur)/1e3 us, AVG(dur)/1e3 avg_us
FROM api GROUP BY 1 ORDER BY us DESC LIMIT 6;
```
```
cudaDeviceSynchronize    2   10498.0 µs   5249.0     <- the final drain, expected
cudaGraphLaunch         32    1501.0 µs     46.9     <- 32 replays, one per token
cudaLaunchKernel       107     318.0 µs      3.0     <- prefill only
cudaMemcpyAsync         32     131.0 µs      4.1
```

```sql
-- 2. Are the kernels tagged with a graph id?
SELECT extract_arg(arg_set_id,'args.graph id') AS graph_id,
       COUNT(*) n, SUM(dur)/1e6 ms
FROM slice WHERE category='kernel' GROUP BY 1 ORDER BY n DESC;
```
```
graph_id=2   n=3424   ms=12.294     <- 32 replays x 107 kernels
graph_id=0   n= 147   ms= 1.184     <- prefill + sampling, not captured
```

`graph id = 0` means "not part of a graph". If your decode kernels show `graph id = 0`, capture silently failed or fell back — common causes are a dynamic shape, a CPU-side branch inside the captured region, or an unsafe op forcing a break.

### 4.3.2 The fan-out trap

One `cudaGraphLaunch` maps to every kernel in the graph:

```sql
SELECT l.name, COUNT(DISTINCT l.id) AS launches, COUNT(*) AS kernels
FROM slice k JOIN flow f ON f.slice_in = k.id JOIN slice l ON f.slice_out = l.id
WHERE k.category='kernel' GROUP BY 1;
```
```
cudaGraphLaunch      32   3424      <- 107:1 fan-out
cudaLaunchKernel    107    107
cuLaunchKernel       32     32
```

Consequently `SUM(ldur)` over `link` counts each graph launch 107 times. The error is enormous and always in the direction of "launches are the bottleneck":

```
naive   SUM(ldur)                  = 160.84 ms      (nonsense)
correct SUM(ldur / launch_fanout)  =   1.75 ms
```

This is why `launch_fanout` is in the prelude. **Any aggregate over launch duration must divide by it.** The same trap applies to `External id` and to any `USING (corr)` join.

### 4.3.3 Measuring the win

The before/after table for the running example is in §6.1 step 8. Two readings
of it belong here.

The GPU did *identical* work in both runs — 12.5 ms of kernels, the same 109 per token. Everything gained came from deleting host time. This is the canonical shape of an inference win, and the reason the first *measurement* in §3.3 is always "what is the GPU utilization", never "which kernel is slowest".

Note also that mean ITL (74 µs) is now *shorter* than per-token GPU time (391 µs): the host runs ahead and the GPU becomes the constraint. The `queue_ns` diagnostic flips from 3.3 µs to 5.2 ms, and utilization reaches 96%. The workload has moved from host-bound to GPU-bound — which means the next optimization is a kernel or quantization change, not another host fix.

### 4.3.4 Residual host work under graphs

```sql
SELECT op, COUNT(*) n, SUM(gdur)/1e3 gpu_us
FROM link WHERE ph='decode' AND (graph_id IS NULL OR graph_id = 0)
GROUP BY 1 ORDER BY gpu_us DESC;
```
```
aten::argmax    32   191.4 µs
aten::copy_     32    26.7 µs
```

Sampling stayed outside the graph. Whether to pull it in is a design decision (it requires a static RNG state and fixed shapes), but the trace quantifies the remaining opportunity precisely.

## 4.4 torch.compile and Inductor kernels

Compiled regions change what you see in the trace in three ways, all of which you should verify rather than assume.

**1. Kernel names become Triton kernels.**
```sql
SELECT CASE WHEN kname GLOB '*triton_*' THEN 'triton'
            WHEN kname GLOB '*cutlass*' OR kname GLOB '*gemm*'
              OR kname GLOB '*gemv*'                        THEN 'cublas/cutlass'
            ELSE 'aten' END AS origin,
       COUNT(*) n, SUM(gdur)/1e3 us
FROM link WHERE ph='decode' GROUP BY 1;
```
Triton kernels are named `triton_poi_fused_add_mul_0`, `triton_red_...` (reduction), `triton_per_...` (persistent). The `fused_<op1>_<op2>_...` suffix names exactly which ATen ops were fused — read it as a fusion receipt.

**2. Kernel count should drop, GPU time may not.** Fusion converts many small elementwise kernels into few larger ones. Compare against the eager baseline (§6.3); the win shows up in *count* and in *host time*, and in the elimination of intermediate-tensor bandwidth.

**3. Recompilation and guards show up as host stalls.** A compiled model that recompiles mid-generation produces a multi-millisecond gap with no CUDA activity. Find it:
```sql
SELECT name, ts, dur/1e6 ms FROM ev
WHERE cat='cpu_op' AND dur > 5000000 ORDER BY dur DESC LIMIT 10;
```
Any multi-ms `cpu_op` in steady-state decode is a recompile, a graph break, or a host-side fallback. Cross-check against `torch._dynamo` logs; in the trace, the signature is a long gap (§3.4.2) with `blocking_call` and `cpu_context` both NULL.

**4. `mode="reduce-overhead"` implies CUDA graphs.** Verify with §4.3.1 — if `graph id = 0` everywhere, the graph path was disabled by a guard failure and you are getting fusion but not launch elimination.


---

[Index](README.md) · [← Chapter 3](chapter-3-performance-model-and-triage.md) · [Chapter 5 →](chapter-5-device-side-bottlenecks.md)
