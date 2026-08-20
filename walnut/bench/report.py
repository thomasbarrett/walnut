"""The tables these benchmarks print, and the caveats that travel with them.

Warnings go to stdout, next to the numbers they qualify: a caveat on stderr
does not survive a copy-paste into a pull request.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from walnut.bench.errors import BenchError
from walnut.bench.metrics import PERCENTILES, SECTIONS


def write_record(record: dict[str, Any], out: str | None) -> None:
    """Dump the record `-o` asked for: everything the table prints, plus the
    full distributions it only digests."""
    if out:
        Path(out).write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {out}")


LABEL = 22
COL = 9


def line(label: str, value: Any, fmt: str = "") -> None:
    """One scalar, value right-aligned so a column of them reads down."""
    print(f"{label:<{LABEL + COL}}{value:>{COL}{fmt}}")


def width(columns: int = 2) -> int:
    """Total width of a table with `columns` numeric columns. The banner is
    drawn to this so it spans what it heads."""
    return LABEL + COL * columns


def rule(title: str = "", columns: int = 2) -> None:
    print(f"{f' {title} ' if title else '':=^{width(columns)}}")


def metrics_table(metrics: dict[str, Any], rows: tuple) -> None:
    """One row per metric, one column per statistic.

    A block per metric buries the comparison that matters — whether the TTFT
    tail is blowing up while TPOT holds — under a scroll. Reading down a p99
    column answers it at a glance.

    `n` is a column because it varies by row: ITL pools every gap of every
    request and reaches thousands where the per-request metrics have hundreds,
    and a percentile means different things at those two sample sizes. A
    percentile whose nearest rank is the sample count is marked, because there
    it *is* the maximum wearing a tail statistic's name.
    """
    present = [(key, label) for key, label in rows if metrics.get(key)]
    columns = ["mean", *(f"p{q:g}" for q in PERCENTILES), "max", "std", "n"]
    print(f"{'':<{LABEL}}" + "".join(f"{c:>{COL}}" for c in columns))

    marked = False
    for key, label in present:
        stats = metrics[key]
        cells = [f"{stats['mean']:.2f}"]
        for q in PERCENTILES:
            cell = f"{stats[f'p{q:g}']:.2f}"
            if not stats.get(f"p{q:g}_resolves", True):
                cell += "*"
                marked = True
            cells.append(cell)
        cells.append(f"{stats['max']:.2f}")
        cells.append(f"{stats['std']:.3f}")
        cells.append(str(stats["samples"]))
        print(f"{label:<{LABEL}}" + "".join(f"{c:>{COL}}" for c in cells))

    if marked:
        print("* rank equals the sample count: this is the maximum, not a tail.")


# -- serve ------------------------------------------------------------------


def report_serve(record: dict[str, Any]) -> None:
    columns = len(PERCENTILES) + 4
    rule("Serving Benchmark Result", columns)
    line("Successful requests:", record["completed"])
    if record["failed"]:
        line("Failed requests:", record["failed"])
    if record["truncated"]:
        line("Truncated mid-stream:", record["truncated"])
    if record["max_concurrency"] is not None:
        line("Max concurrency (client):", record["max_concurrency"])
    line("Concurrency (mean):", record["concurrency"], ".2f")
    line("Concurrency (peak):", record["peak_concurrency"])
    if record["request_rate"] != float("inf"):
        line("Request rate asked (req/s):", record["request_rate"], ".2f")
        line("Request rate achieved (req/s):", record["achieved_rate"], ".2f")
    line("Duration (s):", record["duration_s"], ".2f")
    line("Input tokens:", record["total_input_tokens"])
    line("Generated tokens:", record["total_output_tokens"])
    line("Request throughput (req/s):", record["request_throughput"], ".2f")
    if record["goodput"] is not None:
        line("Request goodput (req/s):", record["goodput"], ".2f")
        line("Goodput (% of requests):", record["goodput_fraction"] * 100, ".1f")
    line("Output throughput (tok/s):", record["output_throughput"], ".2f")
    line("Total throughput (tok/s):", record["total_token_throughput"], ".2f")

    print()
    rows = tuple((key, f"{name} (ms)") for key, name, _ in SECTIONS)
    metrics = dict(record["metrics"])
    # The client-side wait for a --max-concurrency slot is only a row when a
    # limit made the client hold requests back; without one it is all noughts.
    if record["max_concurrency"] is not None:
        metrics["queue_wait"] = record["queue_wait_ms"]
        rows += (("queue_wait", "queue wait (ms)"),)
    metrics_table(metrics, rows)
    rule(columns=columns)
    serve_warnings(record)


def serve_warnings(record: dict[str, Any]) -> None:
    """Every reason the table above might not mean what it says."""
    if not record["token_counts_exact"]:
        print(
            "\n! the server sent no usage chunk, so output tokens were counted "
            "from stream deltas.\n  A delta is not always one token; TPOT is an "
            "upper bound. Check the server's version."
        )
    if record["output_tokens_all"] != [record["max_tokens"]]:
        # Every request is sent with ignore_eos, so every one must return
        # exactly max_tokens. Ragged lengths mean the server ignored the flag
        # or the streams were cut short, and either way the latencies are not
        # what they appear to be.
        raise BenchError(
            f"requests returned {record['output_tokens_all'][:6]} tokens "
            f"rather than {record['max_tokens']}.\n"
            "  Either the server ignored ignore_eos (check its version — an "
            "unknown JSON field is\n  accepted silently) or streams were "
            "truncated. These latencies are not comparable."
        )
    if shortfall(record) > 0.02:
        print(
            f"\n! this client took {record['submit_span_s']:.2f}s to submit a "
            f"schedule that called for {record['scheduled_span_s']:.2f}s "
            f"({shortfall(record) * 100:.0f}% behind).\n"
            "  The load generator was the bottleneck, not the server. "
            "Nothing here describes the engine\n  under the intended load "
            "— re-run with fewer prompts per process, or on a quieter host."
        )
    if record["max_concurrency"] and record["queue_wait_ms"].get("median", 0) > 1:
        print(
            f"\n! requests waited a median "
            f"{record['queue_wait_ms']['median']:.0f} ms for a "
            "--max-concurrency slot.\n  That wait is in this client, not the "
            "engine, and is excluded from TTFT. The server was\n  offered less "
            "load than the arrival rate implies."
        )
    if record["failed"]:
        print(f"\n! {record['failed']} requests failed:")
        for error in record["errors"]:
            print(f"    {error}")


def shortfall(record: dict[str, Any]) -> float:
    """How far behind its own arrival schedule this client fell.

    Separates "the server is slow" from "the harness is slow", and is the rule
    `sweep` stops on. Measured against the schedule, not ``--request-rate``: a
    finite Poisson sample has a realized mean of its own, and charging the
    client for that reports a bottleneck where there is none.
    """
    scheduled = record.get("scheduled_span_s")
    submitted = record.get("submit_span_s")
    if not scheduled or not submitted:
        return 0.0
    return max(0.0, 1 - scheduled / submitted)


# -- sweep ------------------------------------------------------------------


def report_sweep(rungs: list[dict[str, Any]], stopped: str) -> None:
    def tail(record: dict[str, Any], key: str) -> float:
        stats = record["metrics"].get(key) or {}
        return stats.get("p99", 0.0)

    print(f"\n{' Rate Sweep ':=^78}")
    print(
        f"{'rate':>6} {'achieved':>9} {'conc':>6} {'out tok/s':>10} "
        f"{'TTFT p99':>9} {'TPOT p99':>9} {'ITL p99':>8} {'goodput':>8}"
    )
    for record in rungs:
        fraction = record["goodput_fraction"]
        print(
            f"{record['request_rate']:>6g} {record['achieved_rate']:>9.2f} "
            f"{record['concurrency']:>6.1f} {record['output_throughput']:>10.1f} "
            f"{tail(record, 'ttft'):>9.1f} {tail(record, 'tpot'):>9.2f} "
            f"{tail(record, 'itl'):>8.2f} "
            f"{'—' if fraction is None else f'{fraction * 100:.0f}%':>8}"
        )
    print("=" * 78)
    if stopped:
        print(f"\n{stopped}")
    else:
        print(
            "\nevery rung kept up: the knee is above the highest rate offered. "
            "Extend --rates."
        )


# -- latency ----------------------------------------------------------------


def report_latency(record: dict[str, Any]) -> None:
    """The same table `serve` prints. Here the samples are repeats of one
    measurement, so `std` is the noise floor a change has to clear."""
    columns = len(PERCENTILES) + 4
    rule("Latency Benchmark Result", columns)
    line(
        "Iterations (warm-up):", f"{record['num_iters']} ({record['num_iters_warmup']})"
    )
    line("Output tokens:", record["tokens_out"])
    line("Decode tok/s (1/TPOT):", record["tok_per_s"], ".2f")
    line("Output sha:", record["output_sha"])

    print()
    metrics_table(
        record["metrics"],
        tuple((key, f"{name} (ms)") for key, name, _ in SECTIONS),
    )
    rule(columns=columns)
    latency_warnings(record)


def latency_warnings(record: dict[str, Any]) -> None:
    if not record["output_deterministic"]:
        print(
            f"\n! generation is not deterministic across iterations: "
            f"{record['output_sha']}.\n  Per-token metrics are still valid; the "
            "output hash proves nothing."
        )
    if record["hit_eos"]:
        print(
            f"\n! generation stopped at {record['tokens_out']} of "
            f"{record['max_tokens']} tokens (EOS).\n  Per-token metrics are "
            "still valid; e2e is not comparable against a run that generated a "
            "different\n  number of tokens. Pass --ignore-eos."
        )
    if record["num_iters_warmup"] == 0:
        print(
            "\n! --num-iters-warmup 0: the first measured iteration paid for "
            "compilation and graph\n  capture. That is a start-up cost, not a "
            "decode cost — see `walnut bench startup`."
        )


# -- throughput -------------------------------------------------------------


def report_throughput(record: dict[str, Any]) -> None:
    """What the engine does with everything at once."""
    rule("Throughput Benchmark Result")
    line("Successful requests:", record["completed"])
    if record["failed"]:
        line("Failed requests:", record["failed"])
    line("Max batch size:", record["max_batch_size"])
    line("Duration (s):", record["duration_s"], ".2f")
    line("Input tokens:", record["total_input_tokens"])
    line("Generated tokens:", record["total_output_tokens"])
    line("Request throughput (req/s):", record["request_throughput"], ".2f")
    line("Input throughput (tok/s):", record["input_throughput"], ".2f")
    line("Output throughput (tok/s):", record["output_throughput"], ".2f")
    line("Total throughput (tok/s):", record["total_token_throughput"], ".2f")
    rule()
    if record["failed"]:
        print(f"\n! {record['failed']} requests failed:")
        for error in record["errors"]:
            print(f"    {error}")
    if record["output_tokens_all"] != [record["max_tokens"]]:
        print(
            f"\n! requests generated different numbers of tokens "
            f"{record['output_tokens_all'][:6]}, though every one was sent "
            "with ignore_eos.\n  Throughput is still the work the engine did, "
            "but it is not the work you asked for."
        )


# -- startup ----------------------------------------------------------------


def report_startup(record: dict[str, Any]) -> None:
    """Only the cost of getting ready to serve. The moment this table carries
    a tok/s it stops being a start-up measurement."""
    rule("Startup Benchmark Result", 4)
    line(
        "Iterations (warm-up):", f"{record['num_iters']} ({record['num_iters_warmup']})"
    )
    line("Batch slots prepared:", record["max_batch_size"])

    print()
    print(
        f"{'':<{LABEL}}"
        + "".join(f"{c:>{COL}}" for c in ("median", "mean", "std", "n"))
    )
    for key, label in (
        ("load_s", "load weights (s)"),
        ("prepare_s", "prepare batch (s)"),
        ("first_request_s", "first request (s)"),
        ("total_s", "total (s)"),
    ):
        stats = record["phases"].get(key)
        if not stats:
            continue
        cells = (
            f"{stats['median']:.3f}",
            f"{stats['mean']:.3f}",
            f"{stats['std']:.3f}",
            str(stats["samples"]),
        )
        print(f"{label:<{LABEL}}" + "".join(f"{c:>{COL}}" for c in cells))
    rule(columns=4)
    startup_warnings(record)


def startup_warnings(record: dict[str, Any]) -> None:
    if record["num_iters_warmup"] == 0:
        print(
            "\n! --num-iters-warmup 0: the first measured iteration paid for "
            "compilation and graph\n  capture from cold, which is a first-build "
            "cost, not what a restart pays."
        )
