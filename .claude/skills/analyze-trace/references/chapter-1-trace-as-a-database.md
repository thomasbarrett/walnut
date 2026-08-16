# Chapter 1 — The Trace as a Database

> What a Kineto trace contains, how Perfetto's trace processor turns it into tables, and the tools that query them.

## 1.1 Why SQL

A decode step of a small transformer issues ~110 kernels. A 32-token generation issues ~3,500. A 70B model under continuous batching issues millions of events per minute across eight ranks. The three conventional tools all fail at this scale in the same way:

| Tool | Failure mode |
|---|---|
| `prof.key_averages().table()` | Aggregates away time. You learn *what* costs, never *when* or *why the GPU was idle*. |
| Perfetto UI (visual) | Excellent for forming hypotheses on one step. Cannot answer "what is the p99 across 4,000 tokens" or "which of these two traces regressed". |
| Custom JSON parsing | You reimplement interval algebra, slice nesting, and flow resolution — badly, and once per question. |

Trace processor gives you a relational algebra over the trace with the hard parts already solved: hierarchical slice nesting with `ancestor_slice`/`descendant_slice`, causal edges in a `flow` table, and thread/process resolution. Bottleneck analysis is fundamentally a set of **interval questions** — union, gap, containment, overlap — and interval questions are what SQL with window functions is for.

The deeper reason: **inference bottlenecks are rarely in the kernels.** They are in the gaps between them. Aggregate tables cannot see gaps. The trace can, and SQL is how you measure them.

## 1.2 The Kineto trace format

`export_chrome_trace` writes the **Chrome Trace Event Format**: a JSON object whose `traceEvents` array holds one object per event. PyTorch adds a header:

```json
{
  "schemaVersion": 1,
  "deviceProperties": [{"id": 0, "name": "NVIDIA GeForce RTX 5090",
                        "totalGlobalMem": 33711521792, "computeMajor": 12,
                        "numSms": 170, "regsPerMultiprocessor": 65536,
                        "sharedMemPerMultiprocessor": 102400, "warpSize": 32, ...}],
  "cuda_runtime_version": ..., "cupti_version": ..., "record_shapes": true,
  "baseTimeNanoseconds": ..., "traceEvents": [ ... ]
}
```

> **`deviceProperties` is not ingested into any SQL table.** `numSms`, `maxThreadsPerMultiprocessor`, and `sharedMemPerMultiprocessor` are required for wave-quantization and occupancy math (§5.5). Read them out of the JSON header separately and pass them into your analysis as constants. This is the one piece of the trace you must parse yourself.

### 1.2.1 Event phases

| `ph` | Meaning | Where it lands |
|---|---|---|
| `X` | Complete event (has `ts` + `dur`) | `slice` |
| `i` | Instant event | `slice` with `dur = 0` |
| `s` / `f` | Flow start / finish | `flow` |
| `M` | Metadata (`process_name`, `thread_name`) | `process.name`, `thread.name` |

### 1.2.2 Categories — the complete taxonomy

Every event carries a `cat` field. This is the primary key of your mental model. From the real inference trace:

```
$ tp query infer.json "select category, count(*) n, sum(dur)/1e6 ms from slice group by 1 order by n desc"
"cpu_op",              26338,  125.879
"cuda_runtime",         3902,   10.791
"kernel",               3572,   13.517
"user_annotation",        33,   69.935
"gpu_user_annotation",    33,   66.997
"gpu_memcpy",             32,    0.012
"cuda_driver",            32,    0.085
"overhead",                1,    1.298
"Trace",                   1,   70.339
```

| Category | Emitted by | Lives on | Meaning |
|---|---|---|---|
| `cpu_op` | PyTorch dispatcher | host thread | An ATen operator (`aten::mm`) or autograd node. **Nested** — `aten::matmul` contains `aten::mm`. |
| `user_annotation` | `record_function`, `prof.step()` | host thread | Your semantic markers. The root of the host slice tree. |
| `cuda_runtime` | CUPTI | host thread | CUDA Runtime API call (`cudaLaunchKernel`, `cudaStreamSynchronize`). Nested **inside** the `cpu_op` that made it. |
| `cuda_driver` | CUPTI | host thread | Driver API call (`cuLaunchKernel`). cuBLASLt and CUTLASS paths use this instead of the runtime API. **You must handle both.** |
| `kernel` | CUPTI activity | device stream | Actual GPU kernel execution. |
| `gpu_memcpy` | CUPTI activity | device stream | `Memcpy HtoD (Pinned -> Device)`, `DtoH`, `DtoD`. Carries `bytes`. |
| `gpu_memset` | CUPTI activity | device stream | `cudaMemsetAsync` execution. |
| `gpu_user_annotation` | Kineto | device stream | **The GPU-side projection of a `user_annotation`.** Spans from the first to the last device op attributable to that annotation. Enormously useful; see §3.2.3. |
| `ac2g` | Kineto | — | Flow edges: async CPU → GPU. Links a launch to its kernel. |
| `fwdbwd` | Kineto | — | Flow edges linking forward ops to backward nodes. Training only. |
| `cpu_instant_event` | Profiler | host thread | `[memory]` allocation events. |
| `python_function` | `with_stack=True` | host thread | One slice per Python frame; name is `file.py(LINE): func`. |
| `overhead` | Profiler | host thread | Self-reported profiler overhead. |
| `Trace` | Kineto | pseudo-process `Spans` | Outer span for the whole capture. |

### 1.2.3 Arguments, verbatim

**`kernel`:**
```json
{"ph":"X","cat":"kernel","name":"void cutlass::Kernel2<...>(...)",
 "pid":0,"tid":7,"ts":3871627141477.893,"dur":258.498,
 "args":{"queued":0,"device":0,"context":1,"stream":7,"correlation":4732,
         "registers per thread":40,"shared memory":0,
         "blocks per SM":4.705883,"warps per SM":75.294121,
         "grid":[800,1,1],"block":[512,1,1],
         "est. achieved occupancy %":100,
         "occupancy":{"activeBlocksPerMultiprocessor":3,
                      "limitingFactors":"WARPS|REGS","blockLimitRegs":3,
                      "blockLimitSharedMem":100,"blockLimitWarps":3,
                      "allocatedRegistersPerBlock":20480,
                      "allocatedSharedMemPerBlock":1024},
         "graph id":0,"graph node id":0,"channel":0,"channel_type":0}}
```

**`cuda_runtime`:**
```json
{"ph":"X","cat":"cuda_runtime","name":"cudaLaunchKernel","pid":688595,"tid":688595,
 "ts":...,"dur":0.54,"args":{"External id":5,"cbid":317,"correlation":5133}}
```

**`gpu_memcpy`:**
```json
{"args":{"External id":5,"device":0,"context":1,"stream":7,"correlation":5135,
         "bytes":8388608,"memory bandwidth (GB/s)":7.2257,"channel_type":2}}
```

**`cpu_op`** (with `record_shapes`): `External id`, `Record function id`, `Sequence number`, `Fwd thread id`, `Ev Idx`, plus `Input Dims`, `Input Strides`, `Input type`, `Concrete Inputs` as nested arrays.

**`[memory]` instant:** `Total Reserved`, `Total Allocated`, `Bytes` (signed: negative = free), `Device Id`, `Device Type`, `Addr`.

**NCCL kernels** additionally carry: `Collective name`, `dtype`, `In msg nelems`, `Out msg nelems`, `Group size`, `Process Group Name`, `Process Group Ranks`, `Rank`, `Src Rank`, `Dst Rank`, `Comms Id`.

### 1.2.4 The two identifier systems

Understanding these is the difference between a working analysis and a broken one.

**`correlation`** (CUPTI). Set on the host-side API call and on the device activity it produced. This is the **launch → execution** link. Present on `cuda_runtime`/`cuda_driver` and on `kernel`/`gpu_memcpy`/`gpu_memset`.

**`External id`** (PyTorch). Set on the `cpu_op` and propagated to the CUDA API calls made within it. This is the **operator → launch** link. Note it is *not* on the kernel itself.

**Neither is a primary key.** Under CUDA graph replay a single `correlation` covers every kernel in the graph:

```
$ tp query infer_graph.json "with c as (select extract_arg(arg_set_id,'args.correlation') corr
                             from slice where category='kernel')
                             select (select count(*) from c) kernels,
                                    (select count(distinct corr) from c) distinct_corr"
kernels=3571   distinct_corr=179
```

Use the **`flow` table** instead of joining on `correlation` where you can — it is populated from the `ac2g` flow events, is robust to the graph case, and works on ROCm builds where `correlation` is sometimes absent. In non-graph traces the two are equivalent (verified: 899 rows either way).

For operator attribution, prefer **slice ancestry** over `External id`: the CUDA API slice is a *child* of the `cpu_op` in the slice tree, so `ancestor_slice()` recovers the operator directly and gives you the whole call chain for free.

## 1.3 Ingestion semantics: JSON → relational

Trace processor rewrites the JSON into a fixed schema. Six transformations matter.

**1. Timestamps become nanoseconds.** The JSON `ts`/`dur` are floating-point **microseconds**; `slice.ts` and `slice.dur` are integer **nanoseconds**. Divide by `1e3` for µs, `1e6` for ms. Timestamps are absolute (`trace_start()` ≈ 3.87e15 in our trace) — always subtract a reference, never print raw.

**2. Args are prefixed with `args.` and arrays are flattened.**

| JSON | `args.key` |
|---|---|
| `"correlation": 4732` | `args.correlation` |
| `"est. achieved occupancy %": 100` | `args.est. achieved occupancy %` |
| `"grid": [800,1,1]` | `args.grid[0]`, `args.grid[1]`, `args.grid[2]` |
| `"Input Dims": [[1,1024],[1024,4096]]` | `args.Input Dims[0][0]`, `args.Input Dims[0][1]`, `args.Input Dims[1][0]`, … |
| `"occupancy": {"blockLimitRegs": 3}` | `args.occupancy.blockLimitRegs` |

Read them with `extract_arg(arg_set_id, 'args.<key>')`. Forgetting the `args.` prefix silently returns `NULL` — the single most common beginner error. Enumerate what is actually present with:

```sql
SELECT key, COUNT(*) n FROM args GROUP BY 1 ORDER BY n DESC LIMIT 50;
```

**3. Device work is modeled as fake processes and threads.** Kineto encodes GPU activity as `pid = <device_id>`, `tid = <stream_id>`. Trace processor faithfully creates a `process` with `pid=0` and a `thread` named `stream 7`. So:

- **GPU stream** → `thread.name LIKE 'stream %'`, `process.pid = <cuda device>`
- **Host thread** → `process.pid = <real OS pid>`

Do **not** identify device work by `pid = 0`; identify it by `category IN ('kernel','gpu_memcpy','gpu_memset')`. The pid convention collides with the flow-event pseudo-processes and is not stable across Kineto versions.

**4. Track names are NULL.** For JSON traces, `track.name` is empty. All naming lives in `thread.name` / `process.name`, reachable via `thread_track`. Join or use the stdlib:

```sql
INCLUDE PERFETTO MODULE slices.with_context;
SELECT name, thread_name, process_name FROM thread_slice;
```

**5. Slices must nest strictly.** Trace processor enforces a stack discipline per track. Overlapping-but-not-nested complete events are **dropped**, counted in `stats` as `slice_drop_overlapping_complete_event`. See §2.1.3.

**6. Flows become a table.** `s`/`f` event pairs populate `flow(id, slice_out, slice_in)`. On our training trace:

```
out_cat        in_cat        count
cpu_op         cpu_op          546     <- fwdbwd (forward op -> backward node)
cuda_runtime   kernel          528     <- ac2g
cuda_driver    kernel          144     <- ac2g via driver API
cuda_runtime   gpu_memset      183
cuda_runtime   gpu_memcpy       44
```

### 1.3.1 Schema cheat sheet

```
slice(id, ts, dur, track_id, category, name, depth, parent_id, arg_set_id)
thread(utid, tid, name, upid)          process(upid, pid, name)
thread_track(id, utid)                 args(arg_set_id, key, int_value, string_value, real_value)
flow(id, slice_out, slice_in)          stats(name, value, severity, source)
metadata(name, str_value, int_value)
```

Table functions: `ancestor_slice(id)`, `descendant_slice(id)` — both return `slice` rows.
Scalar functions: `extract_arg(arg_set_id, key)`, `trace_start()`, `trace_end()`, `trace_dur()`, `slice_is_ancestor(a, b)`.
Persistence: `CREATE PERFETTO TABLE` (materialized, use for anything joined repeatedly) and `CREATE PERFETTO VIEW` (lazy).

## 1.4 Tooling

Queries in this book are run through
[`scripts/analyze_trace.py`](../scripts/analyze_trace.py), which wraps
`perfetto.trace_processor.TraceProcessor` — the same session, so the prelude
runs once and every later statement sees its views. The `tp` CLI and
`ui.perfetto.dev` speak the same PerfettoSQL if you want a shell or a flame
chart, but nothing here requires them.

### 1.4.1 Recommended workflow

```
capture (§2.1) → preflight (§2.3) → prelude (§2.2) → triage (§3.3) → targeted recipe (Ch. 4–5) → fix → A/B (§6.3)
```


---

[Index](README.md) · [Chapter 2 →](chapter-2-capture-and-view-layer.md)
