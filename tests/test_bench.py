"""Tests for the benchmark harness in `walnut.bench`.

The parts worth testing are the ones that decide whether a number is honest:
the arrival process, the goodput rule, the percentile definition, the sweep's
stopping rule, and the warm-up that keeps a compile time out of a latency.
"""

import dataclasses
import json
import re
import statistics

import pytest
from typer.testing import CliRunner

from walnut.bench import cli as bench_cli
from walnut.bench.errors import BenchError
from walnut.bench.metrics import SLO_METRICS, percentile, summarize
from walnut.bench.online import (
    RequestResult,
    ServeOptions,
    meets_slos,
    peak_concurrency,
    run_serve,
    run_sweep,
)
from walnut.bench.report import report_sweep, shortfall
from walnut.bench.workload import PROMPT, arrival_delays, goodput_config
from walnut.cli import app

runner = CliRunner()

#: Help wraps to the terminal, and a wrapped flag name cannot be searched for.
WIDE = {"COLUMNS": "200"}


def _plain(text: str) -> str:
    """Typer colours its help, and the escape codes land *inside* a flag
    name (`--cuda-graph` prints as `-`, `-cuda`, `-graph` with codes
    between). Strip them before looking for one."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# -- statistics -------------------------------------------------------------


def test_percentile_returns_a_value_that_was_actually_measured():
    """Nearest-rank, not interpolated: a reported p99 is some request's
    latency, not an average of two."""
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 50) == 2.0
    assert percentile(values, 99) == 4.0
    assert percentile(values, 0) == 1.0
    assert percentile([], 99) == 0.0


def test_summarize_carries_its_own_dispersion():
    """`std` travels in the record so no caller can print a value without the
    dispersion it has to be read against."""
    stats = summarize([10.0, 10.0, 12.0], [99.0])
    assert stats["median"] == 10.0
    assert stats["std"] == pytest.approx(0.9428, abs=1e-4)
    assert stats["samples"] == 3
    # p99 of three samples is the maximum wearing a tail statistic's name.
    assert stats["p99_resolves"] is False


def test_std_does_not_grow_with_sample_count():
    """Why dispersion is a standard deviation and not the widest deviation from
    the median.

    Max-deviation is an extreme-value statistic: more samples means more
    chances to stray, so it climbs without settling and two runs at different
    --num-iters could not be read against each other. `std` converges.
    Averaged over trials because a single draw of either is itself noisy.
    """
    import random

    def widest(values: list[float]) -> float:
        mid = statistics.median(values)
        return max(abs(v - mid) for v in values)

    rng = random.Random(0)
    small_std, large_std, small_widest, large_widest = [], [], [], []
    for _ in range(40):
        small = [rng.gauss(100.0, 2.0) for _ in range(20)]
        large = [rng.gauss(100.0, 2.0) for _ in range(400)]
        small_std.append(summarize(small, [])["std"])
        large_std.append(summarize(large, [])["std"])
        small_widest.append(widest(small))
        large_widest.append(widest(large))

    # 20x the samples moves std by a few percent of itself...
    assert statistics.fmean(large_std) == pytest.approx(
        statistics.fmean(small_std), rel=0.15
    )
    # ...and max-deviation by a third, on the very same samples.
    assert statistics.fmean(large_widest) > statistics.fmean(small_widest) * 1.25


# -- the arrival process ----------------------------------------------------


def test_arrival_delays_are_zero_at_an_unlimited_rate():
    import random

    assert arrival_delays(5, float("inf"), 1.0, random.Random(0)) == [0.0] * 5


def test_arrival_delays_average_to_the_configured_rate():
    import random

    delays = arrival_delays(20_000, 4.0, 1.0, random.Random(0))
    assert sum(delays) / len(delays) == pytest.approx(0.25, rel=0.05)


def test_burstiness_above_one_evens_arrivals_out():
    """The knob's whole purpose: same mean rate, different clumping."""
    import random

    bursty = arrival_delays(20_000, 4.0, 0.2, random.Random(0))
    smooth = arrival_delays(20_000, 4.0, 5.0, random.Random(0))
    mean = 0.25
    assert sum(abs(d - mean) for d in bursty) > sum(abs(d - mean) for d in smooth)


# -- goodput ----------------------------------------------------------------


def test_goodput_config_parses_milliseconds_into_seconds():
    assert goodput_config(["ttft:200", "tpot:20"]) == {"ttft": 0.2, "tpot": 0.02}


def test_goodput_config_accepts_one_value_or_many():
    """Repeated or comma-separated, both read the same."""
    assert goodput_config(["ttft:200,tpot:20"]) == goodput_config(
        ["ttft:200", "tpot:20"]
    )


def test_goodput_config_accepts_every_slo_metric():
    assert set(SLO_METRICS) == {"ttft", "tpot", "ntpot", "itl", "e2el"}


def test_goodput_config_rejects_an_unknown_metric():
    with pytest.raises(BenchError):
        goodput_config(["throughput:200"])


def _result(
    *,
    start: float = 0.0,
    ttft: float = 0.05,
    latency: float = 1.0,
    output_tokens: int = 100,
    itl: list[float] | None = None,
) -> RequestResult:
    return RequestResult(
        success=True,
        start=start,
        ttft=ttft,
        latency=latency,
        output_tokens=output_tokens,
        itl=itl or [],
    )


def test_a_request_must_clear_every_slo_not_just_one():
    """A request that answered fast and then stalled served nobody, and each
    metric on its own calls that a success."""
    slos = {"ttft": 0.1, "e2el": 0.5}
    assert meets_slos(_result(latency=0.4), slos)
    assert not meets_slos(_result(latency=2.0), slos)
    assert not meets_slos(_result(ttft=0.5, latency=0.4), slos)


def test_the_itl_slo_is_held_against_the_worst_gap():
    """A single two-second stall is what a user notices; a mean hides it."""
    assert not meets_slos(_result(itl=[0.01] * 99 + [2.0]), {"itl": 0.05})


def test_peak_concurrency_counts_overlapping_requests():
    results = [
        _result(start=0.0, latency=3.0),
        _result(start=1.0, latency=1.0),
        _result(start=1.5, latency=1.0),
        _result(start=10.0, latency=1.0),
    ]
    assert peak_concurrency(results) == 3


def test_tpot_excludes_the_first_token():
    assert _result(ttft=0.1, latency=1.1, output_tokens=101).tpot == pytest.approx(0.01)


def test_ntpot_includes_the_first_token_and_tpot_does_not():
    """The two differ by exactly the prefill, which is why NTPOT is recorded
    and never printed: it cannot tell a stalled prefill from a slow decode."""
    result = _result(ttft=0.5, latency=1.0, output_tokens=100)
    assert result.ntpot == pytest.approx(0.01)
    assert result.tpot == pytest.approx(0.5 / 99)


def test_tpot_is_zero_for_a_single_token_response():
    """There is no inter-token gap to average, and dividing by zero tokens is
    how other harnesses end up reporting TPOT as TTFT."""
    assert _result(output_tokens=1).tpot == 0.0


# -- end to end against a live server ---------------------------------------


def _serve_options(base_url: str, **overrides) -> ServeOptions:
    base = ServeOptions(
        base_url=base_url,
        model=None,
        num_prompts=6,
        request_rate=50.0,
        burstiness=1.0,
        max_concurrency=2,
        dataset="fixed",
        prompt=PROMPT,
        input_len=512,
        range_ratio=0.3,
        tokenizer=None,
        # The stub echoes six words whatever it is asked for, and --ignore-eos
        # is on: a mismatch here is a hard error by design.
        max_tokens=6,
        ignore_eos=True,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        warmups=1,
        goodput={},
        percentiles=[50.0, 99.0],
        timeout=30.0,
        ready_timeout=10.0,
        label="smoke",
        out=None,
    )
    return dataclasses.replace(base, **overrides)


def test_serve_drives_a_live_server_and_reports_exact_token_counts(
    live_server, tmp_path, capsys
):
    """The whole client path: SSE parsing, the usage chunk, the record.

    The stub engine answers in one chunk, so this proves the plumbing rather
    than any latency. What it does prove about numbers is the important part —
    that output tokens come from the server's usage chunk and not from
    counting deltas, which is the difference between a real TPOT and an
    inflated one.
    """
    import asyncio

    out = tmp_path / "record.json"
    opts = _serve_options(live_server, out=str(out))
    assert asyncio.run(run_serve(opts)) == 0

    record = json.loads(out.read_text())
    assert record["mode"] == "serve"
    assert record["completed"] == 6
    assert record["failed"] == 0
    assert record["model"] == "test-model"
    # From the usage chunk, not from counting deltas: the stub echoes the
    # prompt, six words of it, and answers in a single delta.
    assert record["token_counts_exact"] is True
    assert record["total_output_tokens"] == 6 * 6
    assert record["peak_concurrency"] <= 2
    # One metric shape for every metric, in both modes.
    assert set(record["metrics"]) == set(SLO_METRICS)
    assert "Serving Benchmark Result" in capsys.readouterr().out


def test_serve_refuses_rather_than_report_latencies_from_a_dead_server():
    import asyncio

    opts = _serve_options("http://127.0.0.1:9/v1", ready_timeout=1.0)
    with pytest.raises(BenchError, match="not ready"):
        asyncio.run(run_serve(opts))


# -- the sweep's stopping rule ----------------------------------------------


def _rung(rate, achieved, fraction=None, scheduled=10.0, submitted=10.0):
    return {
        "request_rate": rate,
        "achieved_rate": achieved,
        "scheduled_span_s": scheduled,
        "submit_span_s": submitted,
        "goodput_fraction": fraction,
        "concurrency": 4.0,
        "output_throughput": 500.0,
        "metrics": {},
    }


def test_shortfall_is_zero_at_an_unlimited_rate():
    """`inf` submits everything at once: no schedule, nothing to fall behind."""
    assert shortfall(_rung(float("inf"), 12.0, scheduled=0.0)) == 0.0


def test_shortfall_measures_lag_against_the_schedule_not_the_configured_rate():
    """A finite gamma sample path has a realized rate of its own. Judging the
    client against `--request-rate` charges it for the sampler's variance and
    reports a bottleneck at small --num-prompts where there is none."""
    assert shortfall(_rung(24.0, 18.0, scheduled=10.0, submitted=10.0)) == 0.0
    assert shortfall(_rung(24.0, 18.0, scheduled=7.5, submitted=10.0)) == pytest.approx(
        0.25
    )


def test_shortfall_never_goes_negative():
    """Submitting ahead of schedule is not a surplus of anything."""
    assert shortfall(_rung(24.0, 30.0, scheduled=12.0, submitted=10.0)) == 0.0


def test_sweep_reports_the_rung_it_stopped_on(capsys):
    rungs = [_rung(8.0, 7.9, 1.0), _rung(16.0, 15.9, 0.70)]
    report_sweep(rungs, "stopped at 16 req/s: goodput fell to 70%")
    out = capsys.readouterr().out
    assert "Rate Sweep" in out
    assert "70%" in out
    assert "stopped at 16 req/s" in out


def test_sweep_says_so_when_nothing_ever_broke(capsys):
    """A sweep that never found a knee has not found an operating point, and
    reading its top rung as capacity is the mistake it has to prevent."""
    report_sweep([_rung(8.0, 7.9, 1.0)], "")
    assert "the knee is above the highest rate" in capsys.readouterr().out


def test_rates_must_ascend():
    """The sweep stops at the first rung that breaks, which only finds a knee
    if the ladder climbs."""
    import typer

    assert bench_cli._ladder("8,16,24") == [8.0, 16.0, 24.0]
    with pytest.raises(typer.BadParameter, match="ascend"):
        bench_cli._ladder("24,8")


def test_sweep_climbs_a_rate_ladder_and_records_every_rung(
    live_server, tmp_path, capsys
):
    """One record for the whole ladder, so an operating point is chosen from a
    single artifact rather than from several files diffed by eye."""
    import asyncio

    out = tmp_path / "sweep.json"
    opts = _serve_options(
        live_server, num_prompts=4, max_concurrency=None, out=str(out)
    )
    assert asyncio.run(run_sweep(opts, [2.0, 4.0], 0.95)) == 0

    record = json.loads(out.read_text())
    assert record["mode"] == "sweep"
    assert [r["request_rate"] for r in record["rungs"]] == [2.0, 4.0]
    assert all(r["completed"] == 4 for r in record["rungs"])
    assert "Rate Sweep" in capsys.readouterr().out


# -- the command surface ----------------------------------------------------


def test_bench_exposes_the_five_subcommands():
    result = runner.invoke(app, ["bench", "--help"], env=WIDE)
    assert result.exit_code == 0
    for command in ("serve", "throughput", "latency", "startup", "sweep"):
        assert command in _plain(result.stdout)


@pytest.mark.parametrize(
    "command", ["serve", "throughput", "latency", "startup", "sweep"]
)
def test_every_bench_subcommand_takes_a_warmup_flag(command):
    """Warm-up iterations are how every one of these keeps a compile time out
    of its measurement; a subcommand missing the flag has no way to."""
    result = runner.invoke(app, ["bench", command, "--help"], env=WIDE)
    assert result.exit_code == 0
    assert "--num-iters-warmup" in _plain(result.stdout).replace("\n", "")


@pytest.mark.parametrize("command", ["serve", "throughput", "latency", "sweep"])
def test_output_length_is_spelled_the_same_everywhere(command):
    """One name for one quantity. A flag that is --max-tokens under `serve`
    and --output-len under `throughput` is a trap for anyone moving between
    them."""
    result = runner.invoke(app, ["bench", command, "--help"], env=WIDE)
    assert result.exit_code == 0
    help_text = _plain(result.stdout)
    assert "--max-tokens" in help_text
    assert "--output-len" not in help_text


def test_latency_has_no_batch_size_knob():
    """One request in flight is the measurement, not a default. A batch would
    put the scheduler back in it."""
    result = runner.invoke(app, ["bench", "latency", "--help"], env=WIDE)
    assert "--batch-size" not in _plain(result.stdout)
    assert (
        runner.invoke(app, ["bench", "latency", "m", "--batch-size", "8"]).exit_code
        != 0
    )
