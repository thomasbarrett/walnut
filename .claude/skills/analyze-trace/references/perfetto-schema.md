# PerfettoSQL against a torch trace

## Contents

- Units and gotchas
- Tables worth knowing
- Category tags torch emits
- Worked examples

## Units and gotchas

- `ts` and `dur` are **nanoseconds**. Chrome JSON carries microseconds; the
  importer scales them on the way in.
- `dur` is `-1` for a slice that never closed. Filter `dur >= 0`.
- Slices nest. `sum(dur)` grouped by name counts a parent *and* its children,
  so it exceeds the trace span. Subtract children for self time:

  ```sql
  sum(s.dur - coalesce(
      (select sum(c.dur) from slice c where c.parent_id = s.id and c.dur >= 0),
      0))
  ```

- Kernels sit on **thread** tracks. `track.name` is null for them and the
  stream name ("stream 7") lives on `thread.name`, so joining only `track`
  reports a stream as "track 4".

## Tables worth knowing

| Table | Useful columns |
| --- | --- |
| `slice` | `id`, `name`, `category`, `ts`, `dur`, `track_id`, `parent_id`, `arg_set_id` |
| `thread` | `utid`, `tid`, `name` — "stream 7", worker threads |
| `thread_track` | `id`, `utid` — joins a slice's `track_id` to a thread |
| `track` | `id`, `name` |
| `args` | `arg_set_id`, `key`, `int_value`, `string_value` |

`list_tables` equivalents: `select name from perfetto_tables` and
`select name from sqlite_master where type = 'table'`.

## Category tags torch emits

`slice.category` carries torch's own `cat` field:

| Category | What it is |
| --- | --- |
| `cpu_op` | `aten::` operators |
| `kernel` | GPU kernels |
| `gpu_memcpy`, `gpu_memset` | Device transfers and fills |
| `cuda_runtime`, `cuda_driver` | CUDA API calls — see the trap below |
| `python_function` | Python frames (only with `with_stack=True`) |
| `user_annotation` | `record_function` scopes |
| `ac2g` | Flow arrows linking a launch to the kernel it queued |

**The `cuda_runtime` trap.** That category is not "launch cost". It also holds
`cudaDeviceSynchronize`, `cudaStreamSynchronize`, `cudaMemcpyAsync`,
`cudaMalloc`, and `cudaFree`. A synchronize blocks for exactly as long as the
GPU work it waits on, so summing the category makes a kernel-bound run look
launch-bound. Match launches by name (`cudaLaunch*`, `cuLaunch*`,
`cudaGraphLaunch*`) and count only the outermost — a `cuda_driver` call nests
inside the `cuda_runtime` call that made it, so both would count twice.

## Worked examples

Kernels by total time:

```sql
select name, count(*) as calls, sum(dur)/1e3 as us
from slice where category = 'kernel' and dur >= 0
group by name order by us desc limit 20
```

Time by input shape. `args` holds **one row per scalar**, so a shape has to be
reassembled with one join per dimension — grouping by a single dim key silently
merges distinct shapes that happen to share it, and reports one row where there
were two. Shapes are integers, so read `int_value`; `string_value` is null.

```sql
select d00.int_value || 'x' || d01.int_value as lhs,
       d10.int_value || 'x' || d11.int_value as rhs,
       count(*) as calls, sum(s.dur)/1e3 as total_us
from slice s
join args d00 on d00.arg_set_id = s.arg_set_id and d00.key = 'args.Input Dims[0][0]'
join args d01 on d01.arg_set_id = s.arg_set_id and d01.key = 'args.Input Dims[0][1]'
join args d10 on d10.arg_set_id = s.arg_set_id and d10.key = 'args.Input Dims[1][0]'
join args d11 on d11.arg_set_id = s.arg_set_id and d11.key = 'args.Input Dims[1][1]'
where s.name = 'aten::mm' and s.dur >= 0
group by lhs, rhs order by total_us desc
```

The join count follows the operator's arity and rank, so check what keys exist
before writing it:

```sql
select distinct key from args where key glob 'args.Input Dims*' order by key
```

Everything on one stream, in order:

```sql
select s.ts, s.dur, s.name
from slice s
join thread_track tt on tt.id = s.track_id
join thread th using (utid)
where th.name = 'stream 7' and s.dur >= 0
order by s.ts limit 50
```

Gaps between consecutive kernels on a stream:

```sql
select name, gap_us from (
  select s.name,
         (s.ts - lag(s.ts + s.dur) over (partition by s.track_id order by s.ts))
           / 1e3 as gap_us
  from slice s where s.category = 'kernel' and s.dur >= 0
) where gap_us > 100 order by gap_us desc
```
