-- ===========================================================================
-- PerfettoSQL prelude for walnut traces.
--
-- The authoritative definition of the seven relations every query in
-- references/ is written against: ev, dev_op, api, phase, link, kfam, gap.
--
-- `phase` is the one departure. walnut emits no record_function scopes, so
-- `user_annotation` is empty and the book's definition would return no rows,
-- silently emptying every per-token query in Chapters 3 and 6. It reads the
-- Python frames instead, which walnut names for the purpose.
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
-- walnut override: phases from the Python frames walnut names for it.
--
--   prefill              _prefill, once
--   cuda_graph_capture   DecodeGraph.capture, once, with --cuda-graph
--   decode               _decode_step, once per generated token
--
-- Match the name, not the line number in `file(line): function` — the line
-- moves whenever the file above it is edited. Needs the stacks that
-- `walnut profile` records; a server trace has no Python frames, so `phase`
-- comes back empty there.
-- ---------------------------------------------------------------------------
CREATE PERFETTO TABLE phase AS
WITH frame AS (
  SELECT id, ts, dur, ts + dur AS te,
         CASE WHEN name GLOB '*: _prefill'     THEN 'prefill'
              WHEN name GLOB '*: _decode_step' THEN 'decode'
              ELSE 'cuda_graph_capture'
         END AS name
  FROM slice
  WHERE category = 'python_function'
    AND (name GLOB '*: _prefill' OR name GLOB '*: _decode_step'
         OR name GLOB '*graph.py(*): capture')
)
SELECT id, ts, dur, te, name,
       ROW_NUMBER() OVER (PARTITION BY name ORDER BY ts) - 1 AS seq
FROM frame;

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
    -- Attention before gemm: the memory-efficient kernel is named
    -- `fmha_cutlassF_*`, so a '*cutlass*' arm above this one would swallow it
    -- and leave `attention` empty on every trace.
    WHEN kname GLOB '*flash*' OR kname GLOB '*fmha*'           THEN 'attention'
    WHEN kname GLOB '*gemm*' OR kname GLOB '*cutlass*'         THEN 'gemm'
    WHEN kname GLOB '*layer_norm*' OR kname GLOB '*rms_norm*'  THEN 'norm'
    WHEN kname GLOB '*elementwise*'                            THEN 'elementwise'
    WHEN kname GLOB '*reduce*'                                 THEN 'reduce'
    WHEN kname GLOB '*nccl*'                                   THEN 'collective'
    WHEN kname GLOB '*triton_*'                                THEN 'triton'
    ELSE 'other'
  END AS family
FROM link;

-- Reusable gap table over a phase's device ops.
--
-- The running maximum is PARTITIONed by phase. Without it the first gap of each
-- phase is measured from the previous phase's last kernel and charged to this
-- one, which inflates decode idle by the capture-to-decode handover and stops
-- `wall = busy + idle` from closing (~1.4% on a 32-token walnut trace).
-- `seq` rides along so idle can be filtered exactly like busy: summing gaps
-- over all tokens against a `WHERE seq > 0` busy total silently charges the
-- warm-up token's idle to the steady state.
CREATE PERFETTO TABLE gap AS
WITH g AS (SELECT gts, gte, ph, seq FROM link WHERE ph IS NOT NULL),
     m AS (SELECT gts, gte, ph, seq,
                  MAX(gte) OVER (PARTITION BY ph, seq ORDER BY gts
                                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS pm
           FROM g)
SELECT ph, seq, pm AS gap_start, gts AS gap_end, gts - pm AS gap_ns
FROM m WHERE pm IS NOT NULL AND gts > pm;
