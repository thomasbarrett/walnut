# Chapter 2 — Capture and the Analysis Kernel

> Producing a trace you can trust, and the reusable SQL view layer that every later query is written against.

## 2.1 Capturing an inference trace

### 2.1.1 Minimum viable capture

```python
import torch
from torch.profiler import profile, ProfilerActivity, record_function

# 1. WARM UP. Never profile a cold process.
for _ in range(3):
    generate(prompt)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    with torch.inference_mode():
        with record_function("prefill"):
            logits = model(input_ids)
        for i in range(n_tokens):
            with record_function("decode"):
                logits = model(next_tok)
                next_tok = sample(logits)
    torch.cuda.synchronize()          # 2. FLUSH before the profiler stops.
prof.export_chrome_trace("infer.json")
```

Three non-negotiables, in order of how often they are violated:

1. **Warm up.** The first call through any code path pays cuBLAS/cuDNN autotuning, kernel module loading (`cudaFuncSetAttribute`, `cudaMalloc`), and allocator growth. In our trace the first decode token cost 2114 µs against a steady-state 2053 µs; in graph mode the first cost 125 µs against 74 µs — a 70% inflation. Always discard `seq = 0`.
2. **`torch.cuda.synchronize()` inside the `with` block.** CUDA is asynchronous. If the profiler stops while kernels are in flight, CUPTI flushes what it has and the tail of your GPU timeline is silently truncated — you will compute a GPU-busy figure that is too low and diagnose a host-bound workload that isn't.
3. **`record_function` around semantic phases.** The single highest-leverage line of instrumentation in inference profiling. It gives you `user_annotation` slices which become your `GROUP BY` key for everything downstream. Without them you are reduced to inferring token boundaries from kernel patterns.

### 2.1.2 The `record_function` naming convention

Use a **constant** name per phase, not `f"decode_{i}"`. Trace processor will happily store 4,000 distinct strings, but you then cannot group. Recover the index with a window function instead:

```sql
ROW_NUMBER() OVER (PARTITION BY name ORDER BY ts) - 1 AS seq
```

If you need per-token metadata (sequence length, batch size), emit it as a separate instant event or encode it in a second, coarser annotation (`record_function("decode_bs8")`).

### 2.1.3 Capture flags and what they cost

| Flag | Adds | Cost | Use when |
|---|---|---|---|
| `record_shapes=True` | `args.Input Dims[i][j]`, `Input type[i]`, `Concrete Inputs[i]` on every `cpu_op` | ~2–3× trace size; small CPU overhead | Almost always — shapes are how you distinguish gemv from gemm |
| `profile_memory=True` | `[memory]` instant events with `Total Allocated`/`Total Reserved` | ~2 KB per allocation event | KV-cache growth, fragmentation, OOM forensics |
| `with_stack=True` | `python_function` slices (full Python call tree) | 2–4× events; **and see the warning below** | Attributing host time to Python source lines |
| `with_flops=True` | `args.Flops` on supported ops | negligible | Roofline sanity checks |
| `with_modules=True` | module hierarchy in op names | negligible | Layer-level rollups |

> **⚠ `with_stack=True` can silently destroy your phase annotations.**
> With stacks enabled, Kineto emits `python_function` slices that interleave with `user_annotation` slices on the same thread in a way that violates strict nesting. Trace processor's JSON parser **drops** the offending slices. In our capture, all 9 `user_annotation` events vanished from the `slice` table:
>
> ```
> $ tp query train_stack.json "select name,value from stats where value>0 and severity!='info'"
> "slice_drop_overlapping_complete_event",9
> ```
>
> Your `phase` table comes back empty and every per-token query returns zero rows. **Always run the preflight in §2.3.** Workarounds: profile without `with_stack`, or fall back to the GPU-side `gpu_user_annotation` slices (§3.2.4), which live on a different track and survive.

### 2.1.4 Capturing from a serving stack

**vLLM.** Set `VLLM_TORCH_PROFILER_DIR` before starting the server, then toggle capture over the API:

```bash
export VLLM_TORCH_PROFILER_DIR=/traces
vllm serve meta-llama/Llama-3.1-8B &
curl -X POST http://localhost:8000/start_profile
#   ... send a SMALL number of requests ...
curl -X POST http://localhost:8000/stop_profile
```

Related knobs: `torch_profiler_record_shapes`, `torch_profiler_with_stack`, `torch_profiler_with_memory`, `torch_profiler_use_gzip`. Two operational warnings from the vLLM docs: send only a handful of requests (traces explode), and expect the stop call to take minutes to flush — for ~100 requests against a 70B model on H100, roughly 10 minutes.

**Anything else.** If the stack runs PyTorch, `torch.profiler` works; the only question is where to put the `with` block. Wrap the scheduler's step function, not the whole server.

### 2.1.5 Long captures: use the schedule

For steady-state analysis, don't trace everything. `torch.profiler.schedule` gives you a windowed capture:

```python
from torch.profiler import schedule
sched = schedule(wait=5, warmup=2, active=3, repeat=1)
with profile(..., schedule=sched) as prof:
    for req in stream:
        serve(req)
        prof.step()          # drives the state machine
```

`prof.step()` emits a `ProfilerStep#N` `user_annotation` — the training-world equivalent of your `decode` annotation, and equally usable as a `GROUP BY` key. The `warmup` phase runs the profiler machinery without recording, absorbing CUPTI's own start-up cost.

### 2.1.6 Compression

`export_chrome_trace("t.json.gz")` (or vLLM's `use_gzip`) produces a gzipped JSON that trace processor ingests directly:

```bash
$ tp query infer.json.gz "select category, count(*) n from slice group by 1 order by n desc limit 4"
"category","n"
"cpu_op",26338
"cuda_runtime",3902
"kernel",3572
"user_annotation",33
```

Typical ratio is 8–12×. Do this by default for anything you will move between machines.

## 2.2 The view layer

Almost every question in this book is asked against seven relations — `ev`,
`dev_op`, `api`, `phase`, `link`, `kfam`, `gap`. Define them once; everything
downstream is a two-line query. They live in
[`scripts/prelude.sql`](../scripts/prelude.sql), which is the authoritative
copy: read it there rather than from a listing here, because walnut's `phase`
is built from Python frames, not from the `user_annotation` slices this chapter
assumes.

`link` is the one to understand — one row per device op, joined to its
launching API call, its ATen operator, and its phase. Four design decisions in
it are worth stating explicitly:

- **`link` is built from `flow`, not from `correlation`.** Robust to CUDA graphs and to ROCm traces with missing correlation ids.
- **`op` is the *innermost* ancestor** (`ORDER BY depth DESC LIMIT 1`), i.e. `aten::mm`, not the outer `aten::matmul`. Swap to `depth = 0` for the outermost, or drop the `LIMIT` and `group_concat` for the full chain.
- **`ph` is assigned by the *launch* timestamp**, not the kernel's. The kernel may execute long after its phase ended; the host-side launch is what belongs to the phase. Use `ORDER BY p.ts DESC` to pick the innermost annotation when they nest.
- **`launch_fanout`** records how many device ops share one launch. Under CUDA graphs this is 107 in our trace. Any sum over `ldur` must divide by it (§4.3.2).

> **Do not attribute phases by slice ancestry.** It is tempting to use `depth = 0` as the phase, and it works for forward-only inference. It breaks the moment work runs on another thread. In our training trace, the depth-0 ancestor of a backward launch is `autograd::engine::evaluate_function: AddmmBackward0`, not `ProfilerStep#3` — the backward pass runs on a different host thread with its own slice-tree root, so 40% of GPU work was attributed to a phantom "phase". **Attribute temporally.** Anything running off the main thread — autograd, dataloader workers, async schedulers, comms threads — has a different root.

## 2.3 Preflight: validate before you conclude

Run these five checks on every trace before you trust a number.

```sql
-- 1. Did the parser drop anything?
SELECT name, value FROM stats WHERE value > 0 AND severity != 'info';
```
Empty is what you want. `slice_drop_overlapping_complete_event > 0` means your annotations may be gone (§2.1.3).

```sql
-- 2. Is device activity present at all?
SELECT cat, COUNT(*) n, SUM(dur)/1e6 ms FROM ev GROUP BY 1 ORDER BY n DESC;
```
Zero `kernel` rows → CUPTI failed to attach (missing `ProfilerActivity.CUDA`, permissions, or a container without `CAP_SYS_ADMIN`).

```sql
-- 3. Did every device op get linked to a launch?
SELECT (SELECT COUNT(*) FROM dev_op) AS device_ops,
       (SELECT COUNT(*) FROM link)   AS linked;
```
A large shortfall means missing flow events. Fall back to `JOIN ... USING (corr)` and note the caveat in §1.2.4.

```sql
-- 4. Are the phase annotations intact and plausible?
SELECT name, COUNT(*) n, AVG(dur)/1e3 avg_us, MIN(dur)/1e3 min_us, MAX(dur)/1e3 max_us
FROM phase GROUP BY 1;
```

```sql
-- 5. Is the tail truncated? Last device op should not be at the very end of the trace.
SELECT (trace_end() - (SELECT MAX(te) FROM dev_op))/1e6 AS ms_of_trailer;
```
A value near zero suggests the profiler stopped with kernels in flight — go back and add `torch.cuda.synchronize()`.

Real output of check 4 on our two traces:

```
infer.json         decode   32  avg 2055.1 µs   min 1993.5   max 2225.2
                   prefill   1  avg 4171.7 µs
infer_graph.json   decode   32  avg   75.7 µs   min   70.2   max  125.5
                   prefill   1  avg 3306.2 µs
```


---

[Index](README.md) · [← Chapter 1](chapter-1-trace-as-a-database.md) · [Chapter 3 →](chapter-3-performance-model-and-triage.md)
