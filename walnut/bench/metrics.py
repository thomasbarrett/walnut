"""Metric definitions and the statistics over them.

TTFT    request sent -> first content delta
ITL     gap between consecutive content deltas
TPOT    (e2el - ttft) / (output_tokens - 1)
NTPOT   e2el / output_tokens
E2EL    request sent -> last content delta
"""

from __future__ import annotations

import math
import statistics
from typing import Any

DEFAULT_PERCENTILES = "50,90,99"

#: (key, short name, section header, printed).
#:
#: NTPOT is recorded and never printed: dividing the whole request latency by
#: the token count gives a prefill stall and a uniformly slow decode the same
#: value, hiding what ITL exists to show.
METRICS = (
    ("ttft", "TTFT", "Time to First Token", True),
    ("tpot", "TPOT", "Time per Output Token (excl. 1st token)", True),
    ("itl", "ITL", "Inter-token Latency", True),
    ("e2el", "E2EL", "End-to-end Latency", True),
    ("ntpot", "NTPOT", "Normalized Time per Output Token", False),
)

#: Metrics ``--goodput`` accepts.
SLO_METRICS = tuple(key for key, *_ in METRICS)

#: The metrics that get a section in a report.
SECTIONS = tuple((key, name, header) for key, name, header, shown in METRICS if shown)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, so every one reported is a latency something
    actually saw. An interpolated p99 is a number nothing measured."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(q / 100 * len(ordered))
    return ordered[min(len(ordered) - 1, max(0, rank - 1))]


def resolves(q: float, n: int) -> bool:
    """Whether a nearest-rank percentile is distinguishable from the maximum.

    At rank ``ceil(q/100 * n) == n`` the percentile *is* the largest sample —
    one request's latency wearing a tail statistic's name.
    """
    return math.ceil(q / 100 * n) < n


def summarize(values: list[float], percentiles: list[float]) -> dict[str, Any]:
    """One metric's distribution, in ms.

    `std` is the standard deviation, and what it means depends on what the
    samples are. Over repeats of one measurement (`latency`, `startup`) it is
    the noise floor a change has to clear — want about twice it. Over a
    workload of differently-scheduled requests it describes the traffic, and
    the percentiles are what to read.

    Deliberately not the widest deviation from the median: that is an
    extreme-value statistic, so it climbs with sample count and never settles,
    and two runs at different iteration counts could not be read against each
    other.
    """
    if not values:
        return {}
    out: dict[str, Any] = {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "max": max(values),
        "samples": len(values),
    }
    for q in percentiles:
        out[f"p{q:g}"] = percentile(values, q)
        out[f"p{q:g}_resolves"] = resolves(q, len(values))
    return out


def parse_percentiles(value: str) -> list[float]:
    return [float(p) for p in value.split(",") if p.strip()]
