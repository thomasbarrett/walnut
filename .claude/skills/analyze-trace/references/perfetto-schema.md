# PerfettoSQL against a torch trace

Perfetto's `trace_processor` imports a PyTorch Chrome/kineto trace into an
in-memory SQLite database and exposes PerfettoSQL over it: SQLite dialect plus
modules, typed functions, macros, and a small set of table-valued operators.
This is a reference for querying that database — the data model kineto produces,
the language beyond stock SQLite, and the traps that return plausible wrong
numbers rather than errors.

Everything below was verified against traces from `walnut profile
Qwen/Qwen3.5-0.8B --max-tokens 16`, eager and `--cuda-graph`, on an RTX 5090.

---

## 1. Running queries

```bash
uv run python .claude/skills/analyze-trace/scripts/analyze_trace.py sql <trace> < query.sql
```

Or directly, when you want a session that holds state across statements:

```python
from perfetto.trace_processor import TraceProcessor
tp = TraceProcessor(trace="profiles/x.trace.json.gz")
for row in tp.query("select count(*) from slice"):
    print(row.count)
```

Each `tp.query()` runs one statement but shares session state, so `INCLUDE`,
`CREATE PERFETTO TABLE`, and macro definitions persist across calls. A
`CREATE ...` statement returns zero rows; that is success, not failure.

Import is lazy and indexes build on demand. A full-trace self-time query over
134k `cpu_op` slices runs in about 2.5 s.

---

## 2. The data model

Kineto emits Chrome JSON. The importer maps it onto Perfetto's generic
trace model:

| Chrome JSON | Becomes |
| --- | --- |
| `ph:"X"` / `"B"`+`"E"` duration event | a row in `slice` |
| `pid` + `tid` | `process` / `thread`, joined to slices through `thread_track` |
| `cat` | `slice.category` |
| `args` | rows in `args`, keyed by `slice.arg_set_id`, one row per **scalar** |
| `ph:"s"` / `"f"` flow pair (`cat:"ac2g"`) | a row in `flow` — **not** a slice |
| `ph:"M"` metadata | `thread.name`, `process.name` |

Two consequences that shape every query:

- **CPU and GPU work live in the same `slice` table**, separated only by
  `category` and by which thread track they sit on. There is no `gpu_slice`
  content in a torch trace (that view is for Android GPU render stages).
- **`ac2g` is not a category you can select.** The launch→kernel arrows become
  `flow` rows. `select * from slice where category = 'ac2g'` returns nothing.

### Tracks, threads, processes

Kineto puts each CUDA stream on a synthetic process (`pid` = device ordinal)
and a synthetic thread whose name is the stream. So a stream is reached the
same way a CPU thread is — `slice` → `thread_track` → `thread`:

```sql
select th.name as thread, count(*) as n
from slice s
join thread_track tt on tt.id = s.track_id
join thread th using (utid)
group by 1 order by n desc
```

```
thread                  | n
thread 684200 (walnut)  | 257937     -- the CPU thread that opened the window
stream 7                | 38667      -- kernels, memcpy, memset
PyTorch Profiler        | 1
```

`track.name` is **null** for these; the human-readable name is on `thread.name`.
Joining only `track` reports a stream as "track 4".

---

## 3. Units and invariants

- **`ts` and `dur` are nanoseconds.** Chrome JSON carries microseconds; the
  importer scales on the way in. `dur / 1e3` is µs, `dur / 1e6` is ms.
  `dur / 1000` looks like ms and is out by 1000×.
- **`ts` is not zero-based.** It is a raw boot-clock stamp (~3.87e15 here).
  Always take the span from `trace_bounds`, never from `min(ts)` of a filtered
  subset:

  ```sql
  select start_ts, end_ts, (end_ts - start_ts) / 1e6 as span_ms from trace_bounds
  ```

- **`dur = -1` means the slice never closed** (a `B` with no matching `E`).
  Filter `dur >= 0` in every aggregate — `sum()` over a `-1` silently subtracts
  a nanosecond, and `max(dur)` is unaffected while `min(dur)` becomes nonsense.
  An unclosed slice also stays open to the end of its track, so every later
  slice on that thread becomes its *child*, corrupting `depth` and `parent_id`.
  Well-formed `walnut profile` traces have none; traces truncated by a crash or
  by `/stop_profile` racing a step do.
- **Slices nest, so `sum(dur)` double-counts.** Grouping by name sums parents
  and their children alike; the total exceeds the trace span by design. Use
  self time (§7) whenever you rank operators.

---

## 4. Tables and views

The ones that matter for a torch trace:

| Name | Columns worth knowing |
| --- | --- |
| `slice` | `id`, `ts`, `dur`, `track_id`, `category`, `name`, `depth`, `parent_id`, `arg_set_id` |
| `thread` | `utid`, `tid`, `name` |
| `process` | `upid`, `pid`, `name` |
| `thread_track` | `id`, `utid` — joins `slice.track_id` to a thread |
| `args` | `arg_set_id`, `key`, `int_value`, `string_value`, `real_value` |
| `flow` | `id`, `slice_out`, `slice_in`, `trace_id` |
| `trace_bounds` | `start_ts`, `end_ts` |

`slice` also carries `cat` (an alias of `category`) and `slice_id` (an alias of
`id`), plus `thread_ts` / `thread_dur` — thread CPU time, which kineto leaves
null. `perfetto_table_info('slice')` fails; `slice` is a view. To introspect,
use `select * from slice limit 1` or:

```sql
select name from sqlite_master where type in ('table','view') order by name
```

Table-valued functions worth knowing: `ancestor_slice(id)`, `descendant_slice(id)`,
`following_flow(id)`, `preceding_flow(id)`, `directly_connected_flow(id)`.

---

## 5. PerfettoSQL beyond SQLite

### INCLUDE PERFETTO MODULE

Pulls in a stdlib module. Verified present and useful here:

| Module | Provides |
| --- | --- |
| `slices.with_context` | `thread_slice`, `process_slice` — slices pre-joined to thread/process names |
| `slices.hierarchy` | ancestry helpers |
| `intervals.intersect` | `_interval_intersect!` |
| `intervals.overlap` | overlap/concurrency helpers |
| `counters.intervals` | counter-to-interval conversion |

`slices.with_context` alone removes most boilerplate — it replaces the
three-table join above:

```sql
include perfetto module slices.with_context;

select thread_name, category, count(*) as n
from thread_slice where dur >= 0 group by 1, 2 order by n desc
```

Without the `INCLUDE`, `thread_slice` fails with `no such table` — it is a
module export, not a built-in.

### CREATE PERFETTO TABLE / VIEW

`TABLE` materializes and is the right choice for anything you join more than
once; `VIEW` re-runs. Both persist for the session.

```sql
create perfetto table kern as
  select id, ts, dur, name, arg_set_id from slice
  where category = 'kernel' and dur >= 0;
```

### CREATE PERFETTO FUNCTION

Typed, with `$`-prefixed parameters. Types: `LONG`, `DOUBLE`, `STRING`, `BOOL`,
`BYTES`.

```sql
create perfetto function us(x LONG) returns DOUBLE as select $x / 1e3;
select us(sum(dur)) from slice where category = 'kernel' and dur >= 0;
```

A function may also return a table (`RETURNS TABLE(name STRING, n LONG)`).

### CREATE PERFETTO MACRO

Textual substitution before parsing, invoked with a trailing `!`. Use it where a
function cannot go — a table position, a column name, a fragment of syntax.

```sql
create perfetto macro kern() returns TableOrSubquery as
  (select * from slice where category = 'kernel' and dur >= 0);
select count(*) from kern!();
```

### EXTRACT_ARG

Pulls one arg by key without a join, dispatching on value type automatically.
Far more readable than joining `args`, and correct where the value might be real
rather than int:

```sql
select name, extract_arg(arg_set_id, 'args.stream') as stream
from slice where category = 'kernel' limit 5
```

Prefer an explicit `args` join only when you need many keys at once, or when a
null must be distinguished from a missing key.

### _interval_intersect!

Intersects two interval sets — the tool for "how much kernel time fell inside
this window". Both arguments must be **Perfetto tables** with exactly `id`,
`ts`, `dur`; a subquery in the macro position is a syntax error.

```sql
include perfetto module intervals.intersect;

create perfetto table gk as
  select id, ts, dur from slice where category = 'kernel' and dur >= 0;
create perfetto table gl as
  select id, ts, dur from slice where name = 'cudaGraphLaunch' and dur >= 0;

select count(*) from _interval_intersect!((gk, gl), ());
```

### SPAN_JOIN

The older virtual-table equivalent. It rejects tables that share any column name
other than the partition key (`SPAN_JOIN: column id present in both tables`), so
alias everything first. `_interval_intersect!` is the better default.

---

## 6. What torch puts in `category`

Full inventory from a GPU decode trace, with counts from the eager run:

| Category | Count | What it is |
| --- | --- | --- |
| `cpu_op` | 134,245 | `aten::` operators |
| `python_function` | 84,546 | Python frames — only with `with_stack=True` |
| `cuda_runtime` | 38,942 | CUDA runtime API calls (`cuda*`) |
| `kernel` | 37,876 | GPU kernels |
| `gpu_memcpy` | 740 | Device transfers |
| `cuda_driver` | 204 | CUDA driver API calls (`cu*`) |
| `gpu_memset` | 51 | Device fills |
| `overhead` | 8 | Profiler's own cost |
| `user_annotation` | — | `record_function` scopes |
| `Trace` | 1 | The profiling window itself |

`python_function` inclusive time (18.5 s) exceeds the 766 ms trace span by 24×
— it is the whole call stack, counted once per frame. Never sum it.

### The `cuda_runtime` trap

`cuda_runtime` is **not** "launch overhead". Its actual name distribution:

```
cudaLaunchKernel          37576
cudaMemcpyAsync             740
cudaDeviceGetAttribute      187
cudaStreamIsCapturing       120
cudaFuncSetAttribute        102
cudaLaunchKernelExC          96
cudaMemsetAsync              51
cudaStreamGetCaptureInfo     34
cudaStreamSynchronize        33
cudaDeviceSynchronize         2
cudaMalloc                    1
```

A synchronize blocks for exactly as long as the GPU work it waits on — in the
graph run, 18 `cudaStreamSynchronize` calls account for 50 ms against 114 ms of
total `cuda_runtime` time. Summing the category makes a GPU-bound run look
launch-bound. Match launches by name instead:

```sql
select name, count(*) as calls, sum(dur) / 1e3 as us
from slice
where dur >= 0
  and (name glob 'cudaLaunch*' or name glob 'cuLaunch*'
       or name glob 'cudaGraphLaunch*')
group by 1 order by us desc
```

`cuda_driver` is a separate launch path, not a nested duplicate. Its 204 slices
have `cpu_op` (187) or `python_function` (17) parents — never `cuda_runtime` —
because cuBLAS calls `cuLaunchKernel` directly rather than through the runtime.
So counting both categories by name is correct here and does not double-count.
Verify before assuming, with the parent-category query in §8.

---

## 7. Self time

Self time — inclusive minus the children — is the only ranking that makes sense
for nested slices. `aten::linear` otherwise outranks the `aten::addmm` doing the
work.

```sql
select s.name, count(*) as calls,
       sum(s.dur - coalesce(
           (select sum(c.dur) from slice c
            where c.parent_id = s.id and c.dur >= 0), 0)) / 1e3 as self_us
from slice s
where s.category = 'cpu_op' and s.dur >= 0
group by 1 order by self_us desc limit 10
```

```
aten::mul            8896   31378
aten::mm             3179   24415
aten::add            4948   20243
aten::copy_          8542   20172
aten::_to_copy       6715   14678
```

The `dur >= 0` on the **inner** query matters as much as on the outer: one
unclosed child inflates every ancestor's self time.

---

## 8. Args

`args` holds **one row per scalar**. A nested JSON value is flattened with
bracket suffixes — `"grid": [64,1,1]` becomes `args.grid[0]`, `args.grid[1]`,
`args.grid[2]`. There is no way to read the list back in one column.

### On `cpu_op`

`args.External id`, `args.Ev Idx`, `args.Sequence number`, `args.Fwd thread id`,
`args.Record function id`, `args.Python id`, `args.Python parent id`,
`args.Op count`, and — with `record_shapes=True` —
`args.Input Dims[i][j]`, `args.Input Strides[i][j]`, `args.Input type[i]`,
`args.Concrete Inputs[i]`.

### On `kernel`

`args.device`, `args.context`, `args.stream`, `args.correlation`,
`args.External id`, `args.registers per thread`, `args.shared memory`,
`args.blocks per SM`, `args.warps per SM`, `args.est. achieved occupancy %`,
`args.grid[0..2]`, `args.block[0..2]`, `args.queued`, `args.channel`,
`args.channel_type`, `args.graph id`, `args.graph node id`, and the
`args.occupancy.*` family (`activeBlocksPerMultiprocessor`,
`allocatedRegistersPerBlock`, `allocatedSharedMemPerBlock`, `blockLimitBarriers`,
`blockLimitBlocks`, `blockLimitRegs`, `blockLimitSharedMem`, `blockLimitWarps`,
`limitingFactors`).

### On `gpu_memcpy`

Adds `args.bytes` and `args.memory bandwidth (GB/s)`.

### Reading shapes

One join per dimension. Grouping by a single dim key silently merges distinct
shapes that share it and reports one row where there were two. Dims are
integers, so read `int_value`; `string_value` is null.

```sql
select d00.int_value || 'x' || d01.int_value as lhs,
       d10.int_value || 'x' || d11.int_value as rhs,
       count(*) as calls, sum(s.dur) / 1e3 as total_us
from slice s
join args d00 on d00.arg_set_id = s.arg_set_id and d00.key = 'args.Input Dims[0][0]'
join args d01 on d01.arg_set_id = s.arg_set_id and d01.key = 'args.Input Dims[0][1]'
join args d10 on d10.arg_set_id = s.arg_set_id and d10.key = 'args.Input Dims[1][0]'
join args d11 on d11.arg_set_id = s.arg_set_id and d11.key = 'args.Input Dims[1][1]'
where s.name = 'aten::mm' and s.dur >= 0
group by 1, 2 order by total_us desc
```

The join count follows the operator's arity and rank, so check what exists
first — an `aten::mul` on a 4-D tensor has `Input Dims[0][3]`:

```sql
select distinct key from args where key glob 'args.Input Dims*' order by key
```

Parent category, for checking nesting assumptions:

```sql
select p.category as parent_cat, count(*) as n
from slice s join slice p on p.id = s.parent_id
where s.category = 'cuda_driver' group by 1
```

---

## 9. Flows: launch → kernel

Every GPU slice in both traces has exactly one incoming flow — verified:

```sql
select count(*) from slice
where category in ('kernel','gpu_memcpy','gpu_memset')
  and id not in (select slice_in from flow)   -- 0
```

`slice_out` is the launching CPU call, `slice_in` the GPU work:

```sql
select ls.name as launch, ls.category, count(*) as n
from flow f
join slice ks on ks.id = f.slice_in
join slice ls on ls.id = f.slice_out
group by 1, 2 order by n desc
```

Flow beats correlating by `args.correlation` manually: the importer has already
done the matching, and the join is on indexed integer ids.

**Flows fan out under CUDA graphs.** In the graph run, 16 `cudaGraphLaunch`
calls are `slice_out` for 32,464 kernels. Joining flow to kernels therefore
repeats each launch slice thousands of times, and `sum(ls.dur)` over that join
overcounts launch cost by the fan-out factor. Aggregate launch time from the
launch slices directly, never through a flow join.

---

## 10. CUDA graphs

`args.graph id` on a kernel is the discriminator: nonzero means the kernel was
replayed from a captured graph, zero means it was launched eagerly.

```sql
select extract_arg(arg_set_id, 'args.graph id') as graph_id,
       count(*) as kernels, sum(dur) / 1e3 as us
from slice where category = 'kernel' and dur >= 0 group by 1 order by kernels desc
```

```
graph_id | kernels | us
5        | 31792   | 55034     -- replayed from the captured decode graph
0        | 12063   | 21751     -- prefill, warmup, sampling
```

**Graph-replayed kernels carry no `args.External id`.** Only the 12,063 eager
kernels have one. External id is the link back to the `aten::` op that issued
the work, so any attribution of GPU time to operator names covers the eager
kernels only and silently omits the graph body — which is the hot path. Attribute
graph kernels by name, or by their position within the `cudaGraphLaunch` window
via `_interval_intersect!`.

The eager-vs-graph comparison, from these two traces:

| | eager | graph |
| --- | --- | --- |
| `cudaLaunchKernel` | 37,576 | 13,762 |
| `cudaGraphLaunch` | 0 | 16 (one per decoded token) |
| `cuda_runtime` total | 104 ms | 114 ms |
| kernel total | 67.9 ms | 76.8 ms |
| span (`trace_bounds`) | 766 ms | 353 ms |

Launch *count* is what graphs remove, not kernel time. `cuda_runtime` totals
barely move because the graph run's time shifts into `cudaStreamSynchronize`
(the CPU blocked on the GPU) and `cudaGraphLaunch` — which is charged for the
whole replay, not for a launch.

---

## 11. Recipes

Kernels by total device time:

```sql
select name, count(*) as calls, sum(dur) / 1e3 as us
from slice where category = 'kernel' and dur >= 0
group by 1 order by us desc limit 20
```

Device time split by category — what separates "GPU busy" from "GPU computing":

```sql
select category, count(*) as slices, sum(dur) / 1e3 as us
from slice where category in ('kernel','gpu_memcpy','gpu_memset') and dur >= 0
group by 1 order by us desc
```

GPU busy fraction against the real trace span:

```sql
select sum(dur) / 1e6 as busy_ms,
       (select (end_ts - start_ts) / 1e6 from trace_bounds) as span_ms,
       100.0 * sum(dur) / (select end_ts - start_ts from trace_bounds) as busy_pct
from slice where category = 'kernel' and dur >= 0
```

Idle gaps on a stream. Partition by `track_id` — a `lag()` without it walks
across streams and reports gaps that never existed:

```sql
select name, gap_us from (
  select s.name,
         (s.ts - lag(s.ts + s.dur) over (partition by s.track_id order by s.ts)) / 1e3
           as gap_us
  from slice s
  where s.category in ('kernel','gpu_memcpy','gpu_memset') and s.dur >= 0
) where gap_us > 100 order by gap_us desc limit 20
```

Launch cost against device time:

```sql
select
  (select sum(dur) / 1e3 from slice
   where dur >= 0 and (name glob 'cudaLaunch*' or name glob 'cuLaunch*'
                       or name glob 'cudaGraphLaunch*')) as launch_us,
  (select sum(dur) / 1e3 from slice
   where category = 'kernel' and dur >= 0) as kernel_us
```

Time the CPU spends blocked on the GPU:

```sql
select name, count(*) as calls, sum(dur) / 1e3 as us
from slice
where name glob '*Synchronize*' and dur >= 0 group by 1 order by us desc
```

Everything on one stream in order. Note the **trailing space** (§12) — use
`glob`, not `=`:

```sql
select s.ts, s.dur, s.name
from slice s
join thread_track tt on tt.id = s.track_id
join thread th using (utid)
where th.name glob 'stream 7*' and s.dur >= 0
order by s.ts limit 50
```

Splitting prefill from decode. Both live in one trace and have opposite shapes
— prefill runs few large kernels, decode runs thousands of microsecond GEMVs —
so a whole-trace ratio describes neither. Find the boundary from the largest
idle gap or from the first `cudaGraphLaunch`, then:

```sql
select case when ts < <boundary_ns> then 'prefill' else 'decode' end as phase,
       count(*) as kernels, sum(dur) / 1e3 as busy_us
from slice where category = 'kernel' and dur >= 0 group by 1
```

---

## 12. Trap index

Each of these returns a wrong number rather than an error.

1. **`thread.name` for a stream has a trailing space.** It is `'stream 7 '`
   (length 9). `where th.name = 'stream 7'` matches **zero** rows; the same
   query with `glob 'stream*'` matches 38,667. Verified in both traces. Use
   `glob`, `like`, or `trim(th.name)` — never `=`.
2. **`dur / 1000` is not milliseconds.** Nanoseconds: `/1e3` → µs, `/1e6` → ms.
3. **`sum(dur)` grouped by name double-counts** through nesting. Use self time.
4. **`sum(dur)` includes `dur = -1`** for unclosed slices. Always `dur >= 0`.
5. **`category = 'ac2g'` matches nothing** — flow arrows are `flow` rows.
6. **`thread_slice` needs `include perfetto module slices.with_context`.**
7. **`sum(cuda_runtime)` is not launch overhead** — it contains synchronizes,
   which are the *opposite* symptom. Match launch names.
8. **Flow joins fan out under CUDA graphs** — one `cudaGraphLaunch` maps to
   thousands of kernels. Never `sum()` launch duration through a flow join.
9. **Graph-replayed kernels have no `External id`** — operator attribution
   silently skips the hot path.
10. **`lag()` without `partition by track_id`** invents gaps between streams.
11. **`min(ts)` is not the trace start, and neither is the `Trace` slice.**
    The single `category = 'Trace'` slice is kineto's profiling window and runs
    *short* of the imported span — 522 ms against 766 ms eager, 268 ms against
    353 ms with graphs. Denominators come from `trace_bounds`.
12. **Grouping by one `Input Dims` key merges distinct shapes.** Join every
    dimension.
13. **`_interval_intersect!` rejects subqueries** — pass Perfetto tables with
    exactly `id`, `ts`, `dur`.
14. **`SPAN_JOIN` rejects duplicate column names** across its two inputs.
15. **`busy_pct` is per stream** and streams run concurrently, so summing
    across streams can exceed 100%.
16. **A `/start_profile` trace has no `aten::` names.** Kineto records `cpu_op`
    only on the thread that opened the window, so a server-side profile shows
    the CUDA timeline without operator names. Use `walnut profile` when you need
    both.
