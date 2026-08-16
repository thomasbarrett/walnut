# Chapter 5 — Device-Side Bottlenecks

> What to do once the GPU is actually the constraint: kernels, shapes, memory traffic, concurrency, and occupancy.

## 5.1 Kernel inventory

Once §3.3 says GPU-bound (or after you have fixed the host), work the kernels.

### 5.1.1 Top kernels by device time

```sql
SELECT kname, COUNT(*) n, SUM(gdur)/1e3 AS us, AVG(gdur)/1e3 AS avg_us,
       SUM(gdur) * 100.0 / (SELECT SUM(gdur) FROM link WHERE ph='decode') AS pct
FROM link WHERE ph='decode'
GROUP BY 1 ORDER BY us DESC LIMIT 10;
```
```
kernel                                          n     us       avg_us  pct
internal::gemvx::kernel<...>                    800   6756.0   8.44    53.9
pytorch_flash::flash_fwd_splitkv_kernel<...>    256   1484.3   5.80    11.8
vectorized_layer_norm_kernel<...>               544   1278.7   2.35    10.2
elementwise_kernel<128,4,...>                   512    746.0   1.46     6.0
internal::gemvx::kernel<...>                    256    735.7   2.87     5.9
flash_fwd_splitkv_combine_kernel<...>           256    574.1   2.24     4.6
vectorized_elementwise_kernel<4,...>            512    422.9   0.83     3.4
vectorized_elementwise_kernel<4,...>            256    285.4   1.11     2.3
reduce_kernel<512,1,...>                         32    191.6   5.99     1.5
indexSelectSmallIndex<...>                       32     44.3   1.38     0.4
```

Everything you need is here. `gemvx` — CUDA's matrix-*vector* kernel — is 60% of device time across 1,056 invocations: the model is bandwidth-bound on weight reads, exactly as decode theory predicts. Attention is 16% (split-KV flash plus its combine pass). Layer norms and elementwise ops are 22% across 1,824 launches averaging 1–2 µs each — **prime fusion targets**, and the reason `torch.compile` helps decode even when it cannot beat cuBLAS on the GEMMs.

### 5.1.2 Rollup by operator

Kernel names are unreadable template soup. Roll up by the ATen op instead:

```sql
SELECT op, COUNT(*) n, SUM(gdur)/1e3 AS gpu_us,
       SUM(ldur/launch_fanout)/1e3   AS cpu_launch_us
FROM link WHERE ph='decode'
GROUP BY 1 ORDER BY gpu_us DESC LIMIT 10;
```
```
op                                  n      gpu_us    cpu_launch_us
aten::mm                            1056   7491.7    3010.5
aten::_flash_attention_forward       512   2058.4    1349.7
aten::native_layer_norm              544   1278.7    1314.1
aten::copy_                          512    746.0    1368.8
aten::add                            512    422.9    1267.5
aten::silu                           256    285.4     629.6
aten::argmax                          32    191.6      92.8
aten::index_select                    32     44.3     137.6
aten::_local_scalar_dense             32     11.9     171.6
```

The `cpu_launch_us` column is the punchline: for `native_layer_norm`, `copy_`, and `add`, **the host spends more time launching than the GPU spends executing**. Any op where `cpu_launch_us > gpu_us` is pure overhead and must be fused or graphed away.

### 5.1.3 Kernel name normalization

Kernel names carry shapes and template parameters. `kfam` — defined in
[`scripts/prelude.sql`](../scripts/prelude.sql) — strips them down to a
`family` column (`gemv`, `gemm`, `attention`, `norm`, `elementwise`, `reduce`,
`collective`, `triton`, `other`), so:

```sql
SELECT ph, family, COUNT(*) n, SUM(gdur)/1e3 us,
       SUM(gdur)*100.0/SUM(SUM(gdur)) OVER (PARTITION BY ph) AS pct
FROM kfam GROUP BY 1,2 ORDER BY ph, us DESC;
```

This is the rollup to put in a dashboard: it is stable across PyTorch versions and immediately shows a shift in the compute mix (e.g. `gemm` → `triton` after enabling `torch.compile`).

### 5.1.4 The tiny-kernel census

```sql
SELECT CASE WHEN gdur <  1000 THEN 'a <1us'
            WHEN gdur <  2000 THEN 'b 1-2us'
            WHEN gdur <  5000 THEN 'c 2-5us'
            WHEN gdur < 20000 THEN 'd 5-20us'
            ELSE                   'e >20us' END AS bucket,
       COUNT(*) n, SUM(gdur)/1e3 us,
       AVG(ldur/launch_fanout)/1e3 AS avg_launch_us
FROM link WHERE ph='decode' GROUP BY 1 ORDER BY 1;
```

Any bucket where `avg_launch_us` approaches the kernel duration is being paid for twice. In eager decode, 69% of kernels are under 5 µs against a 2.7 µs launch: the launch machinery costs roughly half of what the kernels do, before counting dispatch.

## 5.2 Shapes, dtypes, and the gemv trap

With `record_shapes=True`, `Input Dims` on `cpu_op` lets you tie kernel selection to tensor geometry — the mechanism behind most inference performance cliffs.

```sql
SELECT extract_arg(arg_set_id,'args.Input Dims[0][0]') || 'x' ||
       extract_arg(arg_set_id,'args.Input Dims[0][1]') AS lhs,
       extract_arg(arg_set_id,'args.Input Dims[1][0]') || 'x' ||
       extract_arg(arg_set_id,'args.Input Dims[1][1]') AS rhs,
       COUNT(*) n
FROM slice WHERE name = 'aten::mm' GROUP BY 1,2 ORDER BY n DESC;
```
```
lhs        rhs           n
1x1024     1024x1024     256     <- decode: qkv proj / o proj  (gemv)
1x1024     1024x3072     256     <- decode: fused qkv          (gemv)
1x1024     1024x4096     256     <- decode: mlp up             (gemv)
1x4096     4096x1024     256     <- decode: mlp down           (gemv)
1x1024     1024x32000    33      <- decode: lm_head            (gemv)
512x1024   1024x1024     8       <- prefill                    (gemm)
512x1024   1024x3072     8       <- prefill                    (gemm)
512x1024   1024x4096     8       <- prefill                    (gemm)
```

The `M = 1` rows are the entire decode story. At `M = 1` there is no data reuse across rows: every weight element is read once and used once, arithmetic intensity is ~1 FLOP/byte, and no tensor core can help. cuBLAS dispatches `gemvx` rather than a tensor-op GEMM, which is why the prefill kernels are `cutlass_80_tensorop_f16_s16816gemm` and the decode kernels are not.

**The actionable consequence:** decode throughput is bounded by `model_bytes / HBM_bandwidth`, and the only lever that changes the *shape* is batch size.

Neither input is in the trace, so fetch them before quoting the bound:

- `model_bytes` — the weights **actually read to produce one token**, times
  bytes per element for the serving dtype (2 for fp16/bf16, 1 for fp8/int8).
  That is every transformer-block weight plus `lm_head`. **Exclude the input
  embedding table**: decode reads a single row of it, not the matrix. On a
  small model that matters — a 0.8B with vocab 150k × hidden 1024 carries ~19%
  of its parameters in embeddings, so a naive `total_params × dtype_bytes`
  overstates `model_bytes` and understates the ceiling by the same fraction.
  When embeddings are *tied*, `lm_head` is that same matrix and is read in
  full: count it once. Get the count from `model.safetensors.index.json`, the
  model card, or `sum(p.numel() for p in model.parameters())` — not from
  `config.json`, which carries architecture dims and `torch_dtype`, never a
  parameter count.
- `HBM_bandwidth` — `nvidia-smi --query-gpu=name --format=csv` and then the
  card's spec sheet. Use ~80% of the theoretical peak as the achievable figure;
  a well-written gemv reaches roughly that.

The ceiling is `HBM_bandwidth / model_bytes` tokens/s at batch 1 **and short
context**. Once the context is long enough to matter, per-token traffic is
`model_bytes + 2 × n_layers × n_kv_heads × head_dim × seq_len × dtype_bytes`
for the KV read (§5.3).

Quote the ceiling next to the measured rate — the ratio is the headroom, and it
is the number worth arguing over. Then check whether the weight reads are
actually the problem: the gemv family's achieved bandwidth is
`model_bytes / SUM(gdur)` over one steady-state token,

```sql
SELECT SUM(gdur)/1e3 AS gemv_us_per_token
FROM kfam WHERE family = 'gemv' AND ph = 'decode' AND seq = 1;
```

If that comes out near peak while the overall ratio is poor, the loss is not in
the weight reads, and no amount of kernel tuning on the GEMMs will recover it —
look at what surrounds them (§5.1.1) and at the host (Chapter 4).

Confirm the batch-size transition empirically:

```sql
-- Correlate M with the kernel family actually chosen.
SELECT extract_arg(s.arg_set_id,'args.Input Dims[0][0]') AS M,
       k.family, COUNT(*) n, AVG(k.gdur)/1e3 avg_us
FROM kfam k
JOIN slice s ON s.id = (SELECT a.id FROM ancestor_slice(k.lid) a ORDER BY a.depth DESC LIMIT 1)
WHERE s.name IN ('aten::mm','aten::addmm','aten::bmm')
GROUP BY 1,2 ORDER BY M;
```

Run this across traces at batch 1, 8, 32, 64 and you get the batch size at which your stack starts using tensor cores — the single most important number for a throughput-oriented deployment. Also check dtype:

```sql
SELECT extract_arg(arg_set_id,'args.Input type[0]') dtype, COUNT(*) n
FROM slice WHERE category='cpu_op' AND name IN ('aten::mm','aten::addmm') GROUP BY 1;
```
An unexpected `float` among `c10::Half` rows means a silent upcast — usually a `LayerNorm`, a norm-weight, or a KV cache left in fp32.

## 5.3 Memory traffic and the KV cache

```sql
SELECT kname, COUNT(*) n, SUM(gdur)/1e3 us,
       SUM(bytes)/1e6 AS mb,
       AVG(extract_arg(gargs,'args.memory bandwidth (GB/s)')) AS gbs
FROM link WHERE gcat = 'gpu_memcpy' GROUP BY 1 ORDER BY us DESC;
```
On the decode trace, the answer is almost nothing — which is the correct answer:
```
Memcpy DtoH (Device -> Pinned)    32     11.9 µs   4 B each    (the sampled token)
```
A workload that *does* move data per step (here, a training loop uploading a batch every iteration) looks like this:
```
Memcpy HtoD (Pinned -> Device)     3   3484.5 µs   25.17 MB    7.22 GB/s
Memcpy DtoD (Device -> Device)    51    185.8 µs  427.82 MB   2307.15 GB/s
Memcpy DtoH (Device -> Pinned)     2      2.1 µs    0.00 MB      0.00 GB/s
```

How to read the three directions:

- **HtoD** — input upload. 7.2 GB/s here against a PCIe Gen5 x16 ceiling well above that: the transfers are small enough to be latency-dominated. Check pinning and `non_blocking=True`; unpinned memory forces a staging copy and blocks. In decode this should be nearly zero — if you see per-token HtoD, you are uploading something you should keep resident (position ids, masks, rotary tables).
- **DtoD** — 428 MB at 2.3 TB/s. This is KV-cache writes and layout shuffles. Large DtoD volume in decode usually means the cache is being *reallocated or reshaped* rather than written in place; paged/pre-allocated caches should show writes as kernels, not memcpys.
- **DtoH** — logits and sampled tokens. Small volume but latency-critical; see §4.2.

### 5.3.1 Allocator behaviour

With `profile_memory=True`:

```sql
SELECT MAX(extract_arg(arg_set_id,'args.Total Allocated'))/1e6 AS peak_alloc_mb,
       MAX(extract_arg(arg_set_id,'args.Total Reserved'))/1e6  AS peak_reserved_mb,
       COUNT(*) AS n_events
FROM slice WHERE name = '[memory]';
```
```
peak_alloc_mb=1326.7   peak_reserved_mb=1524.6   n_events=1968
```

`reserved − allocated` is caching-allocator slack: 198 MB, ~15%, healthy. Persistent growth in that gap across tokens is fragmentation. Per-token allocation churn is the KV-cache smell to look for:

```sql
WITH p AS (SELECT ts, te, seq FROM phase WHERE name='decode')
SELECT p.seq,
       COUNT(*)                                                          AS n_alloc_events,
       SUM(CASE WHEN extract_arg(m.arg_set_id,'args.Bytes') > 0
                THEN extract_arg(m.arg_set_id,'args.Bytes') END)/1e6     AS alloc_mb,
       MAX(extract_arg(m.arg_set_id,'args.Total Reserved'))/1e6          AS reserved_mb
FROM slice m, p
WHERE m.name='[memory]' AND m.ts >= p.ts AND m.ts < p.te
GROUP BY 1 ORDER BY 1;
```

Rising `reserved_mb` across tokens means the allocator is growing mid-generation — a latency spike waiting to happen, since `cudaMalloc` is synchronizing. Pre-allocate the KV cache to `max_seq_len` at startup.

## 5.4 Streams, overlap, and collectives

### 5.4.1 Stream census

```sql
SELECT stream, gcat, COUNT(*) n, SUM(gdur)/1e6 ms,
       MIN(gts) first_ts, MAX(gte) last_ts
FROM link GROUP BY 1,2 ORDER BY ms DESC;
```

A single-stream trace (`stream = 7` only, as in our example) means **zero overlap is possible**: every kernel waits for its predecessor. Multi-stream traces let you measure actual concurrency.

### 5.4.2 Measuring real overlap

Sum-of-durations minus union-of-intervals gives overlapped time:

```sql
WITH k AS (SELECT gts, gte FROM link ORDER BY gts),
     m AS (SELECT gts, gte, MAX(gte) OVER (ORDER BY gts
                            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS pm FROM k)
SELECT (SELECT SUM(gdur) FROM link)/1e6                                     AS sum_ms,
       SUM(CASE WHEN pm IS NULL OR gts >= pm THEN gte - gts
                ELSE MAX(0, gte - pm) END)/1e6                              AS union_ms,
       ((SELECT SUM(gdur) FROM link)
        - SUM(CASE WHEN pm IS NULL OR gts >= pm THEN gte - gts
                   ELSE MAX(0, gte - pm) END))/1e6                          AS overlapped_ms
FROM m;
```

`overlapped_ms ≈ 0` with multiple streams means your streams are serialized by events or by the default stream. For inference this matters most for: prefill/decode overlap in chunked-prefill schedulers, KV-cache transfers on a copy stream, and speculative-decoding draft/verify pipelines.

### 5.4.3 Collectives (tensor / pipeline parallel)

NCCL kernels appear as `kernel` slices named `ncclDevKernel_AllReduce_*`, `ncclDevKernel_ReduceScatter_*`, etc., carrying rich args.

```sql
SELECT kname,
       extract_arg(gargs,'args.Collective name')  AS collective,
       extract_arg(gargs,'args.dtype')            AS dtype,
       extract_arg(gargs,'args.In msg nelems')    AS in_elems,
       extract_arg(gargs,'args.Group size')       AS group_size,
       COUNT(*) n, SUM(gdur)/1e3 us, AVG(gdur)/1e3 avg_us
FROM link WHERE kname GLOB '*nccl*' GROUP BY 1,2,3,4,5 ORDER BY us DESC;
```

Then the two questions that matter in tensor-parallel decode:

**Share of the token spent in collectives** — in TP decode this is often 20–40% and is the argument for larger TP-friendly fusions or for reducing TP degree:
```sql
SELECT ph,
       SUM(CASE WHEN kname GLOB '*nccl*' THEN gdur ELSE 0 END) * 100.0 / SUM(gdur) AS pct_collective
FROM link GROUP BY 1;
```

**Rank skew** — load a trace per rank into separate trace processor instances and compare per-rank collective *wait*. A rank that arrives early sits inside the collective kernel; the slowest rank sets the pace. In a single-rank trace, the proxy is the variance of collective duration for a fixed message size: high variance at constant `In msg nelems` means you are measuring other ranks' straggling, not the network.

> The queries in §5.4.3 are written against the documented Kineto NCCL argument schema; unlike every other query in this book, they were not executed against a multi-GPU trace on the test machine. Verify the arg keys with the `SELECT key, COUNT(*) FROM args` census (§1.3) before relying on them.

## 5.5 Occupancy and wave quantization

For GPU-bound workloads, the kernel args expose enough to reason about SM utilization without leaving the trace. Recall `numSms` comes from the JSON header (170 on our RTX 5090), not from SQL.

```sql
SELECT kname, COUNT(*) n, SUM(gdur)/1e3 us,
       extract_arg(gargs,'args.grid[0]')
     * extract_arg(gargs,'args.grid[1]')
     * extract_arg(gargs,'args.grid[2]')                     AS blocks,
       extract_arg(gargs,'args.blocks per SM')               AS blocks_per_sm,
       extract_arg(gargs,'args.est. achieved occupancy %')   AS occ_pct,
       extract_arg(gargs,'args.registers per thread')        AS regs,
       extract_arg(gargs,'args.shared memory')               AS smem
FROM link WHERE gcat='kernel' GROUP BY 1 ORDER BY us DESC LIMIT 10;
```

Three readings:

**Under-filled grids.** `blocks < numSms` means part of the GPU is idle for the entire kernel. This is endemic in decode. Measured on our inference trace (170 SMs):

```
kernel                          n     us      blocks  active_bpm  waves  waste
vectorized_elementwise_kernel   561  1333.5   512     12          0.25   75%
elementwise_kernel<128,4,...>   528   777.3   1024    12          0.50   50%
internal::gemvx::kernel<...>    256   735.7   256      3          0.50   50%
```

Not one of these kernels fills a single wave. The elementwise kernels launch 512 blocks where the device can hold 2,040 concurrently — 75% of the machine is idle for their entire duration. This is the hardware-level statement of the same fact §5.2 made at the shape level: at batch size 1 there is not enough parallelism to fill the GPU, and the fix is more batch or more fusion, not a better kernel.

**Wave quantization.** With `blocks_per_wave = numSms × activeBlocksPerMultiprocessor`, the waste is:
```sql
-- substitute your own numSms
WITH cfg(num_sms) AS (VALUES (170))          -- from deviceProperties in the JSON header
SELECT kname, n, us, blocks, active_bpm,
       CAST(blocks AS REAL) / (num_sms * active_bpm)                      AS waves,
       1.0 - CAST(blocks AS REAL)
           / (CEIL(CAST(blocks AS REAL)/(num_sms*active_bpm)) * num_sms * active_bpm) AS waste_frac
FROM (SELECT kname, COUNT(*) n, SUM(gdur)/1e3 us,
             extract_arg(gargs,'args.grid[0]') * extract_arg(gargs,'args.grid[1]')
           * extract_arg(gargs,'args.grid[2]')                             AS blocks,
             extract_arg(gargs,'args.occupancy.activeBlocksPerMultiprocessor') AS active_bpm
      FROM link WHERE gcat='kernel' GROUP BY 1), cfg
WHERE active_bpm > 0 AND blocks > num_sms      -- ignore trivially small launches
ORDER BY waste_frac * us DESC LIMIT 10;
```
A kernel at 1.05 waves wastes ~48% of the second wave; a kernel at 0.25 waves wastes 75% of its only one. Order by `waste_frac * us` so the ranking reflects time actually lost, not the worst ratio on a trivial launch. Landing tile sizes on a wave boundary is a frequently large win for prefill GEMMs; for decode, sub-one-wave kernels are a batching problem, not a tuning problem.

**Occupancy limiters.** `args.occupancy.limitingFactors` is a string like `"WARPS|REGS"`, with `blockLimitRegs`, `blockLimitSharedMem`, `blockLimitWarps` giving the numbers. Weight it by duration so you see what limits the kernels that matter:

```sql
SELECT extract_arg(gargs,'args.occupancy.limitingFactors') AS limiter,
       COUNT(*) n, SUM(gdur)/1e3 us
FROM link WHERE gcat='kernel' GROUP BY 1 ORDER BY us DESC;
```

Caveat: `est. achieved occupancy %` is a *theoretical* figure derived from launch geometry, not a measurement. It is 0 or NULL for some kernels (flash-attention in our trace). High occupancy is neither necessary nor sufficient for high throughput — a well-tuned GEMM at 17% occupancy can saturate the machine. Use it to explain a slow kernel, never to condemn a fast one.


---

[Index](README.md) · [← Chapter 4](chapter-4-host-side-bottlenecks.md) · [Chapter 6 →](chapter-6-practice.md)
