-- ===========================================================================
-- PerfettoSQL prelude for walnut traces.
--
-- The first half is Appendix B of references/ verbatim: ev, dev_op, api,
-- phase, link, kfam, gap.
--
-- The second half overrides `phase` and rebuilds `link`. walnut calls no
-- torch.profiler.record_function, so `user_annotation` is empty and the
-- book's phase table would have zero rows — taking every per-token query in
-- Chapters 3 and 6 down with it. Phases are reconstructed from `aten::item`
-- instead: walnut's sampler pulls each token to the host with
-- `int(next_token.item())`, so there is exactly one per generated token.
--
-- Pipe this in front of any query:
--   cat prelude.sql q.sql | uv run python scripts/analyze_trace.py sql <trace>
-- ===========================================================================

INCLUDE PERFETTO MODULE slices.with_context;

-- Every slice with thread/process context resolved.
CREATE PERFETTO VIEW ev AS
SELECT s.id, s.ts, s.dur, s.ts + s.dur AS te, s.name, s.category AS cat,
       s.track_id, s.parent_id, s.depth, s.arg_set_id,
       th.utid, th.tid, th.name AS thread_name, p.pid, p.name AS process_name
FROM slice s
JOIN thread_track tt ON s.track_id = tt.id
JOIN thread th USING (utid)
JOIN process p USING (upid);

-- Device-side work: kernels, memcpys, memsets.
CREATE PERFETTO VIEW dev_op AS
SELECT *,
       extract_arg(arg_set_id, 'args.stream')      AS stream,
       extract_arg(arg_set_id, 'args.correlation') AS corr,
       extract_arg(arg_set_id, 'args.graph id')    AS graph_id,
       extract_arg(arg_set_id, 'args.bytes')       AS bytes
FROM ev
WHERE cat IN ('kernel', 'gpu_memcpy', 'gpu_memset');

-- Host-side CUDA calls. BOTH runtime and driver APIs.
CREATE PERFETTO VIEW api AS
SELECT *, extract_arg(arg_set_id, 'args.correlation') AS corr
FROM ev
WHERE cat IN ('cuda_runtime', 'cuda_driver');

-- ---------------------------------------------------------------------------
-- walnut override: phases from aten::item, one per generated token.
--
-- prefill    trace start -> the first token reaching the host
-- decode[i]  token i -> token i+1   (so N items yield N-1 decode intervals)
--
-- If a future walnut wraps its phases in record_function, delete this block:
-- the book's own definition will then be correct.
-- ---------------------------------------------------------------------------
CREATE PERFETTO TABLE phase AS
WITH tok AS (
  SELECT ts + dur AS te, ROW_NUMBER() OVER (ORDER BY ts) - 1 AS n
  FROM slice WHERE name = 'aten::item'
),
-- LAG must run over the unfiltered set: SQLite applies WHERE before window
-- functions, so filtering n > 0 first leaves the token-0 -> token-1 interval
-- with a NULL start and drops it from every average.
iv AS (SELECT te, n, LAG(te) OVER (ORDER BY te) AS prev FROM tok)
SELECT 0 AS id,
       (SELECT start_ts FROM trace_bounds) AS ts,
       (SELECT MIN(te) FROM tok) - (SELECT start_ts FROM trace_bounds) AS dur,
       (SELECT MIN(te) FROM tok) AS te,
       'prefill' AS name, 0 AS seq
UNION ALL
SELECT 1000 + n, prev, te - prev, te, 'decode', n - 1
FROM iv WHERE n > 0;

-- The central relation: device op x launching API call x ATen op x phase.
-- Built on `flow`, so it survives CUDA graphs and missing correlation ids.
CREATE PERFETTO TABLE link AS
SELECT
  d.id AS gid, d.ts AS gts, d.dur AS gdur, d.te AS gte, d.name AS kname,
  d.cat AS gcat, d.stream, d.graph_id, d.bytes,
  a.id AS lid, a.ts AS lts, a.dur AS ldur, a.te AS lte, a.name AS lname,
  a.tid AS ltid, d.arg_set_id AS gargs, a.arg_set_id AS largs,
  d.ts - a.te                       AS queue_ns,       -- <0/~0 => GPU starved
  COUNT(*) OVER (PARTITION BY a.id) AS launch_fanout,  -- >1 under CUDA graphs
  (SELECT x.name FROM ancestor_slice(a.id) x ORDER BY x.depth DESC LIMIT 1) AS op,
  (SELECT p.name FROM phase p
    WHERE a.ts >= p.ts AND a.ts < p.te ORDER BY p.ts DESC LIMIT 1)          AS ph,
  (SELECT p.seq  FROM phase p
    WHERE a.ts >= p.ts AND a.ts < p.te ORDER BY p.ts DESC LIMIT 1)          AS seq
FROM flow f
JOIN dev_op d ON f.slice_in  = d.id
JOIN api    a ON f.slice_out = a.id;

-- Kernel family classification for stable rollups.
CREATE PERFETTO VIEW kfam AS
SELECT *,
  CASE
    WHEN kname GLOB '*gemv*'                                   THEN 'gemv'
    WHEN kname GLOB '*gemm*' OR kname GLOB '*cutlass*'         THEN 'gemm'
    WHEN kname GLOB '*flash*' OR kname GLOB '*fmha*'           THEN 'attention'
    WHEN kname GLOB '*layer_norm*' OR kname GLOB '*rms_norm*'  THEN 'norm'
    WHEN kname GLOB '*elementwise*'                            THEN 'elementwise'
    WHEN kname GLOB '*reduce*'                                 THEN 'reduce'
    WHEN kname GLOB '*nccl*'                                   THEN 'collective'
    WHEN kname GLOB '*triton_*'                                THEN 'triton'
    ELSE 'other'
  END AS family
FROM link;

-- Reusable gap table over a phase's device ops.
CREATE PERFETTO TABLE gap AS
WITH g AS (SELECT gts, gte, ph FROM link),
     m AS (SELECT gts, gte, ph,
                  MAX(gte) OVER (ORDER BY gts
                                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS pm
           FROM g)
SELECT ph, pm AS gap_start, gts AS gap_end, gts - pm AS gap_ns
FROM m WHERE pm IS NOT NULL AND gts > pm;
