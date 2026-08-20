"""Metric definitions and the statistics over them.

TTFT    request sent -> first content delta
ITL     gap between consecutive content deltas
TPOT    (e2el - ttft) / (output_tokens - 1)
E2EL    request sent -> last content delta
"""

from __future__ import annotations

import math
import statistics
from typing import Any

#: Fixed, not configurable. The table is built around three columns, and a run
#: reported at other percentiles is comparable with no other run. A knob here
#: also invites a p99.9 over five samples, which is what `resolves` exists to
#: catch — the harness should not hand out a way around its own guardrail.
PERCENTILES = (50.0, 90.0, 99.0)

#: (key, short name, section header).
#:
#: Every metric here is printed. NTPOT — whole-request latency over token
#: count — used to be recorded and withheld, because it gives a stalled prefill
#: and a uniformly slow decode the same value and so cannot show what ITL
#: exists to show. A number that must never be read is not a measurement; the
#: reason it was rejected is the part worth keeping, and this is it.
METRICS = (
    ("ttft", "TTFT", "Time to First Token"),
    ("tpot", "TPOT", "Time per Output Token (excl. 1st token)"),
    ("itl", "ITL", "Inter-token Latency"),
    # Per request this is arithmetic: e2el == ttft + tpot * (tokens - 1), by
    # the definition of TPOT, and output length is held fixed. Its *tail* is
    # not — p99 E2EL cannot be recovered from p99 TTFT and p99 TPOT, because a
    # request can be bad at one without the other. It earns its row as a tail
    # statistic, and the mean can never disagree with the two rows above it.
    ("e2el", "E2EL", "End-to-end Latency"),
)

#: Metrics ``--goodput`` accepts.
SLO_METRICS = tuple(key for key, *_ in METRICS)

#: The metrics that get a section in a report.
SECTIONS = METRICS


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


def summarize(values: list[float]) -> dict[str, Any]:
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
    for q in PERCENTILES:
        out[f"p{q:g}"] = percentile(values, q)
        out[f"p{q:g}_resolves"] = resolves(q, len(values))
    return out
