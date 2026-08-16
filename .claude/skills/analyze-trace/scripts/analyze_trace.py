#!/usr/bin/env python3
"""Query the Chrome traces `walnut profile` writes.

Perfetto's ``trace_processor`` loads a trace into a SQL database; this wraps
the queries that are easy to get wrong. Run a subcommand for those, and `sql`
for anything else — the schema is documented in
``references/perfetto-schema.md``.

``ts`` and ``dur`` are nanoseconds: Chrome JSON carries microseconds and the
importer scales them on the way in. Output columns say which unit they use.

``slice.category`` carries torch's own ``cat`` tag, which is the only thing
separating device work from the CPU that queued it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Protocol

#: Categories torch tags device work with.
DEVICE = ("kernel", "gpu_memcpy", "gpu_memset")

#: Categories holding CUDA API calls. Both appear when the driver and the
#: runtime are traced, and a driver call nests inside the runtime call that
#: made it — so anything summing these must exclude the nested copy.
CUDA_API = ("cuda_runtime", "cuda_driver")

#: Name globs for the launch APIs inside `CUDA_API`.
#:
#: The category also holds cudaDeviceSynchronize, cudaMemcpyAsync, cudaMalloc
#: and friends. A synchronize blocks for as long as the work it waits on, so
#: counting one as launch cost makes a kernel-bound run look launch-bound —
#: the opposite of the truth. Hence matching by name, not by category.
LAUNCH_GLOBS = ("cudaLaunch*", "cuLaunch*", "cudaGraphLaunch*", "cuGraphLaunch*")

#: Name glob for the blocking calls, reported apart as CPU-waits-on-GPU.
SYNC_GLOB = "*Synchronize*"

#: Rows a subcommand prints unless asked for more. Enough to see the shape of
#: a profile without burying the reader; --limit raises it.
DEFAULT_LIMIT = 25
#: Ceiling on rows from one call, so a stray --limit can't dump the trace.
MAX_LIMIT = 500

#: Gaps shorter than this are scheduling noise between consecutive kernels
#: rather than a stall worth chasing.
DEFAULT_MIN_GAP_US = 100.0

# torch puts kernels on thread tracks, where "stream 7" is the *thread's* name
# and the track's own name is null. Reporting a stream as "track 4" would be
# useless, so both device queries resolve the name the same way.
_TRACK_JOIN = (
    "join track t on s.track_id = t.id "
    "left join thread_track tt on tt.id = s.track_id "
    "left join thread th on th.utid = tt.utid"
)
_TRACK_NAME = "coalesce(th.name, t.name, 'track ' || s.track_id)"


class Queryable(Protocol):
    """The part of ``TraceProcessor`` these analyses use."""

    def query(self, sql: str) -> Any: ...


# --- helpers ---------------------------------------------------------------


def _rows(tp: Queryable, sql: str) -> list[dict[str, Any]]:
    """Run ``sql`` and materialize it as plain dicts."""
    return [dict(row.__dict__) for row in tp.query(sql)]


def _clamp(limit: float) -> int:
    """Bound a row limit, coercing so `limit 5.5` can't reach the SQL."""
    return max(1, min(int(limit), MAX_LIMIT))


def _quote(values: tuple[str, ...] | list[str]) -> str:
    """Render a SQL string list.

    Doubling the quote is complete escaping in SQLite, which is what
    PerfettoSQL is: there is no backslash escape to work around.
    """
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def _is_launch(alias: str) -> str:
    """SQL predicate selecting kernel-launch calls on ``alias``."""
    globs = " or ".join(f"{alias}.name glob {_quote([g])}" for g in LAUNCH_GLOBS)
    return f"{alias}.category in ({_quote(CUDA_API)}) and ({globs})"


def trace_span_ns(tp: Queryable) -> int:
    """Wall-clock nanoseconds from the first slice to the last."""
    row = _rows(
        tp, "select min(ts) as start, max(ts + dur) as end from slice where dur >= 0"
    )[0]
    return int((row["end"] or 0) - (row["start"] or 0))


# --- analyses --------------------------------------------------------------


def overview(tp: Queryable) -> dict[str, Any]:
    """Summarize a trace: span, slice categories, threads, device work.

    Per-category and per-thread totals are *inclusive* — a nested slice counts
    in its own row and again in every ancestor's — so they exceed the trace
    span and are named ``inclusive_us`` to say so. `top_operators` subtracts
    children where an exclusive figure is wanted.
    """
    span_ns = trace_span_ns(tp)
    slices = _rows(tp, "select count(*) as n from slice where dur >= 0")[0]["n"]

    categories = _rows(
        tp,
        "select coalesce(category, '(none)') as category, count(*) as slices, "
        "sum(dur) / 1e3 as inclusive_us from slice where dur >= 0 "
        "group by category order by inclusive_us desc",
    )
    threads = _rows(
        tp,
        "select coalesce(th.name, 'tid ' || th.tid) as thread, count(*) as slices, "
        "sum(s.dur) / 1e3 as inclusive_us from slice s "
        "join thread_track tt on s.track_id = tt.id "
        "join thread th using (utid) where s.dur >= 0 "
        "group by thread order by inclusive_us desc limit 20",
    )
    device_slices = sum(
        int(row["slices"]) for row in categories if row["category"] in DEVICE
    )
    return {
        "duration_ms": round(span_ns / 1e6, 3),
        "slices": slices,
        "has_device_work": device_slices > 0,
        "device_slices": device_slices,
        "categories": categories,
        "threads": threads,
    }


def top_operators(
    tp: Queryable,
    limit: int = DEFAULT_LIMIT,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Rank slice names by self time — ``key_averages()``, but queryable.

    Self time is a slice's duration less its direct children's, summed per
    name. Without it a parent is credited with the work its children did, and
    ``aten::linear`` outranks the ``aten::addmm`` doing the arithmetic.
    """
    where = ["s.dur >= 0"]
    if category:
        where.append(f"s.category = {_quote([category])}")
    return _rows(
        tp,
        f"""
        select s.name as name,
               coalesce(s.category, '(none)') as category,
               count(*) as calls,
               sum(s.dur) / 1e3 as inclusive_us,
               sum(s.dur - coalesce(
                   (select sum(c.dur) from slice c
                    where c.parent_id = s.id and c.dur >= 0), 0)) / 1e3 as self_us,
               avg(s.dur) / 1e3 as avg_us
        from slice s
        where {" and ".join(where)}
        group by s.name, s.category
        order by self_us desc
        limit {_clamp(limit)}
        """,
    )


def device_utilization(
    tp: Queryable,
    min_gap_us: float = DEFAULT_MIN_GAP_US,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Report per-stream GPU busy time and the largest idle gaps.

    ``busy_pct`` is measured against the whole trace, not the stretch between
    the first and last kernel: idle before the first kernel and after the last
    is real idle time, and a prefill phase or a cold pass would otherwise
    vanish from the denominator.

    Busy time stays per track. Kernels on one stream cannot overlap, so a
    per-track sum is exact, while summing across streams would double-count
    work that genuinely ran at the same time.
    """
    device_in = _quote(DEVICE)
    span_ms = trace_span_ns(tp) / 1e6
    tracks = _rows(
        tp,
        f"""
        select s.track_id as track_id,
               {_TRACK_NAME} as track,
               count(*) as slices,
               sum(s.dur) / 1e6 as busy_ms,
               (max(s.ts + s.dur) - min(s.ts)) / 1e6 as active_span_ms
        from slice s {_TRACK_JOIN}
        where s.category in ({device_in}) and s.dur >= 0
        group by s.track_id, track
        order by busy_ms desc
        """,
    )
    for track in tracks:
        busy = track["busy_ms"] or 0
        track["busy_pct"] = round(100 * busy / span_ms, 2) if span_ms else 0

    gaps = _rows(
        tp,
        f"""
        select track, gap_us, ts_ns, before_slice, after_slice from (
          select {_TRACK_NAME} as track,
                 (s.ts - lag(s.ts + s.dur) over w) / 1e3 as gap_us,
                 s.ts as ts_ns,
                 lag(s.name) over w as before_slice,
                 s.name as after_slice
          from slice s {_TRACK_JOIN}
          where s.category in ({device_in}) and s.dur >= 0
          window w as (partition by s.track_id order by s.ts)
        )
        where gap_us > {float(min_gap_us)}
        order by gap_us desc
        limit {_clamp(limit)}
        """,
    )
    return {
        "trace_span_ms": round(span_ms, 3),
        "tracks": tracks,
        "idle_gaps": gaps,
        "note": (
            "No device slices — this is a CPU-only trace."
            if not tracks
            else "busy_pct is per track, against the whole trace span."
        ),
    }


def launch_overhead(tp: Queryable, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """Compare time spent launching kernels against time running them.

    This is the number ``--cuda-graph`` moves: replaying a captured graph drops
    the per-token launch cost. A ``launch_per_device_pct`` near the device
    total means the loop is launch-bound.

    Only the launch APIs are counted, and only the outermost of them — a driver
    call nested in the runtime call that made it would otherwise count twice.
    Blocking calls are reported apart as ``synchronize_ms``: those are the CPU
    waiting on the GPU, which is the reverse problem.
    """
    launch, device_in = _is_launch("s"), _quote(DEVICE)
    outermost = (
        f"not exists (select 1 from slice p "
        f"where p.id = s.parent_id and ({_is_launch('p')}))"
    )
    counted = f"({launch}) and s.dur >= 0 and {outermost}"
    sync = (
        f"s.category in ({_quote(CUDA_API)}) "
        f"and s.name glob {_quote([SYNC_GLOB])} and s.dur >= 0"
    )
    totals = _rows(
        tp,
        f"""
        select
          (select coalesce(sum(s.dur), 0) / 1e6 from slice s
           where {counted}) as launch_ms,
          (select count(*) from slice s where {counted}) as launches,
          (select coalesce(sum(s.dur), 0) / 1e6 from slice s
           where {sync}) as synchronize_ms,
          (select coalesce(sum(dur), 0) / 1e6 from slice
           where category in ({device_in}) and dur >= 0) as device_ms
        """,
    )[0]
    calls = _rows(
        tp,
        f"""
        select s.name as name, count(*) as calls, sum(s.dur) / 1e3 as total_us,
               avg(s.dur) / 1e3 as avg_us
        from slice s where {counted}
        group by s.name order by total_us desc limit {_clamp(limit)}
        """,
    )
    launch_ms = totals["launch_ms"] or 0
    device_ms = totals["device_ms"] or 0
    return {
        "launch_ms": round(launch_ms, 3),
        "device_ms": round(device_ms, 3),
        "synchronize_ms": round(totals["synchronize_ms"] or 0, 3),
        "launches": totals["launches"],
        "launch_per_device_pct": (
            round(100 * launch_ms / device_ms, 2) if device_ms else None
        ),
        "by_call": calls,
    }


def run_query(tp: Queryable, sql: str, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """Run arbitrary PerfettoSQL.

    ``limit`` is applied to the results rather than injected into ``sql``, so
    aggregates and CTEs mean what they say; a truncated result reports it.
    """
    rows = _rows(tp, sql)
    capped = _clamp(limit)
    return {
        "columns": list(rows[0]) if rows else [],
        "rows": rows[:capped],
        "row_count": len(rows),
        "truncated": len(rows) > capped,
    }


# --- rendering -------------------------------------------------------------


def _cell(value: Any) -> str:
    """Format one value: floats to 1dp, None as an empty cell."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def render_table(rows: list[dict[str, Any]]) -> str:
    """Render rows as an aligned table.

    A table beats JSON here: the output goes into a context window, and the
    braces and repeated keys of JSON are pure overhead when reading.
    """
    if not rows:
        return "(no rows)"
    columns = list(rows[0])
    cells = [[_cell(row.get(column)) for column in columns] for row in rows]
    widths = [
        max(len(column), *(len(row[index]) for row in cells))
        for index, column in enumerate(columns)
    ]
    lines = ["  ".join(c.ljust(w) for c, w in zip(columns, widths, strict=True))]
    lines.append("  ".join("-" * w for w in widths))
    lines.extend(
        "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in cells
    )
    return "\n".join(line.rstrip() for line in lines)


def render_overview(result: dict[str, Any]) -> str:
    device = "yes" if result["has_device_work"] else "no"
    return "\n".join(
        [
            f"duration_ms: {result['duration_ms']}   slices: {result['slices']}   "
            f"device work: {device} ({result['device_slices']} slices)",
            "",
            "categories (inclusive of nested slices):",
            render_table(result["categories"]),
            "",
            "threads:",
            render_table(result["threads"]),
        ]
    )


def render_device(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"trace_span_ms: {result['trace_span_ms']}",
            result["note"],
            "",
            "per stream:",
            render_table(result["tracks"]),
            "",
            "largest idle gaps:",
            render_table(result["idle_gaps"]),
        ]
    )


def render_launch(result: dict[str, Any]) -> str:
    pct = result["launch_per_device_pct"]
    verdict = (
        "no device work in this trace" if pct is None else f"{pct}% of device time"
    )
    return "\n".join(
        [
            f"launch_ms: {result['launch_ms']} over {result['launches']} launches "
            f"({verdict})",
            f"device_ms: {result['device_ms']}   "
            f"synchronize_ms: {result['synchronize_ms']} (CPU blocked on GPU)",
            "",
            "by call:",
            render_table(result["by_call"]),
        ]
    )


def render_query(result: dict[str, Any]) -> str:
    table = render_table(result["rows"])
    if result["truncated"]:
        table += f"\n({result['row_count']} rows, showing {len(result['rows'])})"
    return table


# --- CLI -------------------------------------------------------------------


def open_trace(path: str) -> Any:
    """Open a trace, failing with a usable message rather than a traceback."""
    trace = Path(path).expanduser()
    if not trace.is_file():
        sys.exit(f"error: no trace at {trace}")
    try:
        from perfetto.trace_processor import TraceProcessor
    except ImportError:
        sys.exit(
            "error: the 'perfetto' package is missing. Run this through the "
            "project environment: uv run python <this script> ... "
            "(after uv sync --extra cpu --dev)"
        )
    try:
        return TraceProcessor(trace=str(trace))
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as text
        sys.exit(f"error: could not read {trace} as a trace: {exc}")


def read_sql(args: argparse.Namespace) -> str:
    """Take SQL from --sql, a file, or stdin — stdin avoids shell quoting."""
    if args.sql:
        return args.sql
    if args.sql_file:
        if args.sql_file == "-":
            return sys.stdin.read()
        return Path(args.sql_file).read_text()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    sys.exit("error: no SQL given. Use --sql, --sql-file, or pipe it on stdin.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_trace.py",
        description="Query a Chrome trace written by `walnut profile`.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON, not a table.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        child = sub.add_parser(name, help=help_text, description=help_text)
        child.add_argument("trace", help="Path to a .trace.json.gz file.")
        # Accepted after the subcommand too, which is where it reads naturally.
        # SUPPRESS so an unused child flag doesn't clobber the parent's value.
        child.add_argument(
            "--json",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Emit JSON, not a table.",
        )
        return child

    add("overview", "Span, slice categories, threads, and whether there is GPU work.")

    ops = add("top-ops", "Rank operations by self time, exclusive of children.")
    ops.add_argument(
        "--category",
        help="Restrict to one category: kernel, cpu_op, python_function, ...",
    )
    ops.add_argument("--limit", type=int, default=DEFAULT_LIMIT)

    device = add("device", "GPU busy time per stream and the largest idle gaps.")
    device.add_argument("--min-gap-us", type=float, default=DEFAULT_MIN_GAP_US)
    device.add_argument("--limit", type=int, default=DEFAULT_LIMIT)

    launch = add("launch", "Kernel-launch cost against kernel execution time.")
    launch.add_argument("--limit", type=int, default=DEFAULT_LIMIT)

    sql = add("sql", "Run PerfettoSQL. Reads stdin when --sql is not given.")
    sql.add_argument("--sql", help="Query text. Beware shell quoting; prefer stdin.")
    sql.add_argument("--sql-file", help="Read the query from a file, or '-' for stdin.")
    sql.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # Read stdin before opening the trace: trace_processor is slow to start,
    # and a missing query should fail immediately.
    sql = read_sql(args) if args.command == "sql" else None

    tp = open_trace(args.trace)
    try:
        result: Any
        if args.command == "overview":
            result = overview(tp)
            text = render_overview(result)
        elif args.command == "top-ops":
            result = top_operators(tp, limit=args.limit, category=args.category)
            text = render_table(result)
        elif args.command == "device":
            result = device_utilization(
                tp, min_gap_us=args.min_gap_us, limit=args.limit
            )
            text = render_device(result)
        elif args.command == "launch":
            result = launch_overhead(tp, limit=args.limit)
            text = render_launch(result)
        else:
            result = run_query(tp, sql or "", limit=args.limit)
            text = render_query(result)
    finally:
        # trace_processor runs as a subprocess and has no finalizer; without
        # this it outlives the interpreter, still holding the whole trace.
        tp.close()

    print(json.dumps(result, indent=2) if args.json else text)


if __name__ == "__main__":
    main()
