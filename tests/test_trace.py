import gzip
import io
import json
import pathlib

import analyze_trace as analysis
import pytest
import torch

from walnut.profiler import TorchProfiler


class FakeQueryable:
    """Records the SQL it is handed and replays canned rows."""

    def __init__(self, rows=()):
        self.sql: list[str] = []
        self._rows = list(rows)

    def query(self, sql):
        self.sql.append(sql)
        return [type("Row", (), {"__dict__": dict(r)})() for r in self._rows]


def write_trace(path, events):
    """Write ``events`` as the gzipped Chrome JSON torch exports."""
    with gzip.open(path, "wt") as handle:
        json.dump({"traceEvents": events}, handle)
    return path


def open_trace(path):
    processor = pytest.importorskip(
        "perfetto.trace_processor",
        reason="perfetto is a dev dependency",
    )
    return processor.TraceProcessor(trace=str(path))


@pytest.fixture(scope="module")
def trace_path(tmp_path_factory):
    """A real torch trace, so the SQL runs against torch's own output."""
    directory = tmp_path_factory.mktemp("profiles")
    profiler = TorchProfiler(directory, with_stack=False)
    profiler.start()
    layer = torch.nn.Linear(64, 64)
    x = torch.randn(8, 64)
    for _ in range(5):
        x = torch.relu(layer(x))
    return profiler.stop().trace


@pytest.fixture(scope="module")
def tp(trace_path):
    handle = open_trace(trace_path)
    yield handle
    handle.close()


# --- limits ----------------------------------------------------------------


def test_limits_are_capped():
    fake = FakeQueryable()
    analysis.top_operators(fake, limit=10_000)
    assert f"limit {analysis.MAX_LIMIT}" in fake.sql[0]


def test_limits_are_at_least_one():
    fake = FakeQueryable()
    analysis.top_operators(fake, limit=0)
    assert "limit 1" in fake.sql[0]


def test_limits_are_coerced_to_integers():
    """A client sending 5.5 would otherwise reach SQL as `limit 5.5`."""
    assert analysis._clamp(5.5) == 5


def test_a_quoted_category_cannot_break_out(tp):
    """Doubling the quote is complete escaping in SQLite."""
    assert analysis.top_operators(tp, category="cpu_op' or '1'='1") == []


def test_run_query_truncates_without_touching_the_sql():
    """A LIMIT injected into the SQL would change what aggregates mean."""
    fake = FakeQueryable([{"n": i} for i in range(10)])
    result = analysis.run_query(fake, "select 1 as n", limit=3)
    assert fake.sql == ["select 1 as n"]
    assert result["row_count"] == 10
    assert result["truncated"] is True
    assert len(result["rows"]) == 3
    assert result["columns"] == ["n"]


def test_run_query_reports_no_truncation():
    fake = FakeQueryable([{"n": 1}])
    assert analysis.run_query(fake, "select 1 as n", limit=3)["truncated"] is False


def test_run_query_on_an_empty_result():
    result = analysis.run_query(FakeQueryable(), "select 1 where 0")
    assert result == {
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
    }


# --- against a real trace --------------------------------------------------


def test_overview_reads_a_real_trace(tp):
    result = analysis.overview(tp)
    assert result["duration_ms"] > 0
    assert result["slices"] > 0
    categories = {row["category"] for row in result["categories"]}
    assert "cpu_op" in categories  # torch's tag for aten ops
    assert result["threads"]


def test_overview_reports_a_cpu_trace_as_deviceless(tp):
    """The CPU fixture has no kernels; the device tools would be empty."""
    result = analysis.overview(tp)
    assert result["has_device_work"] is False
    assert result["device_slices"] == 0


def test_top_operators_finds_the_matmul(tp):
    rows = analysis.top_operators(tp, limit=analysis.MAX_LIMIT, category="cpu_op")
    assert "aten::addmm" in [row["name"] for row in rows]
    assert all(row["category"] == "cpu_op" for row in rows)


def test_top_operators_ranks_by_self_time(tp):
    """The matmul must outrank the wrapper that merely contains it."""
    rows = analysis.top_operators(tp, limit=analysis.MAX_LIMIT, category="cpu_op")
    order = {row["name"]: index for index, row in enumerate(rows)}
    assert order["aten::addmm"] < order["aten::linear"]
    # Self time is exclusive, so it cannot exceed the inclusive total.
    for row in rows:
        assert row["self_us"] <= row["inclusive_us"] + 1e-6


def test_top_operators_is_sorted(tp):
    rows = analysis.top_operators(tp, limit=10)
    assert rows == sorted(rows, key=lambda r: r["self_us"], reverse=True)
    assert len(rows) <= 10


def test_device_utilization_on_a_cpu_trace(tp):
    """It must say so rather than divide by zero."""
    result = analysis.device_utilization(tp)
    assert result["tracks"] == []
    assert result["idle_gaps"] == []
    assert "CPU-only" in result["note"]


def test_launch_overhead_on_a_cpu_trace(tp):
    result = analysis.launch_overhead(tp)
    assert result["device_ms"] == 0
    assert result["launch_per_device_pct"] is None  # not a division by zero


def test_query_runs_perfetto_sql(tp):
    result = analysis.run_query(tp, "select count(*) as n from slice")
    assert result["rows"][0]["n"] > 0


# --- against GPU-shaped traces ---------------------------------------------
#
# CI has no GPU, so the device analyses would otherwise never see a kernel.
# torch exports plain Chrome JSON, so a hand-built trace takes the same import
# path. Timestamps are microseconds, as in torch's export.


@pytest.fixture(scope="module")
def gpu_tp(tmp_path_factory):
    """Four kernels on one stream, with a 550us stall in the middle."""
    events = []
    for index in range(4):
        events.append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "pid": 1,
                "tid": 1,
                "ts": 100 + index * 100,
                "dur": 10,
            }
        )
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": f"sgemm_{index % 2}",
                "pid": 1,
                "tid": 7,
                "ts": 200 + index * 100 + (500 if index >= 2 else 0),
                "dur": 50,
            }
        )
    events.append(
        {
            "ph": "M",
            "name": "thread_name",
            "pid": 1,
            "tid": 7,
            "args": {"name": "stream 7"},
        }
    )
    path = write_trace(tmp_path_factory.mktemp("gpu") / "gpu.trace.json.gz", events)
    handle = open_trace(path)
    yield handle
    handle.close()


@pytest.fixture(scope="module")
def bound_tp(tmp_path_factory):
    """A GPU-bound run: 10us launches, 1000us kernels, and the CPU blocked in
    cudaDeviceSynchronize for as long as each kernel runs.

    Counting a synchronize as launch cost would report this — 50us of launches
    against 5000us of kernels — as launch-bound, which is backwards.
    """
    events = []
    for index in range(5):
        base = index * 2000
        events.append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "pid": 1,
                "tid": 1,
                "ts": base,
                "dur": 10,
            }
        )
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "sgemm",
                "pid": 1,
                "tid": 7,
                "ts": base + 20,
                "dur": 1000,
            }
        )
        events.append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaDeviceSynchronize",
                "pid": 1,
                "tid": 1,
                "ts": base + 20,
                "dur": 1000,
            }
        )
    path = write_trace(tmp_path_factory.mktemp("bound") / "b.trace.json.gz", events)
    handle = open_trace(path)
    yield handle
    handle.close()


def test_overview_sees_device_work(gpu_tp):
    result = analysis.overview(gpu_tp)
    assert result["has_device_work"] is True
    assert result["device_slices"] == 4


def test_device_utilization_measures_busy_time(gpu_tp):
    (track,) = analysis.device_utilization(gpu_tp, min_gap_us=10)["tracks"]
    assert track["busy_ms"] == pytest.approx(0.2)  # 4 kernels x 50us


def test_busy_pct_is_measured_against_the_whole_trace(gpu_tp):
    """Idle before the first kernel and after the last is still idle."""
    result = analysis.device_utilization(gpu_tp, min_gap_us=10)
    (track,) = result["tracks"]
    span = result["trace_span_ms"]
    assert span == pytest.approx(0.95)  # ts 100us to 1050us
    assert track["busy_pct"] == pytest.approx(100 * 0.2 / 0.95, abs=0.1)
    # The stretch between the first and last kernel is shorter, and using it
    # as the denominator would overstate utilization.
    assert track["active_span_ms"] < span


def test_device_utilization_names_the_stream(gpu_tp):
    """Kernels sit on a thread track, so the name comes off the thread."""
    (track,) = analysis.device_utilization(gpu_tp, min_gap_us=10)["tracks"]
    assert track["track"] == "stream 7"


def test_device_utilization_finds_the_stall(gpu_tp):
    gaps = analysis.device_utilization(gpu_tp, min_gap_us=10)["idle_gaps"]
    assert gaps[0]["gap_us"] == pytest.approx(550.0)
    assert gaps == sorted(gaps, key=lambda g: g["gap_us"], reverse=True)


def test_device_utilization_honours_the_gap_floor(gpu_tp):
    """The 50us gaps are noise next to the 550us stall."""
    gaps = analysis.device_utilization(gpu_tp, min_gap_us=100)["idle_gaps"]
    assert [g["gap_us"] for g in gaps] == [pytest.approx(550.0)]


def test_top_operators_filters_to_kernels(gpu_tp):
    rows = analysis.top_operators(gpu_tp, category="kernel")
    assert {row["name"] for row in rows} == {"sgemm_0", "sgemm_1"}
    assert all(row["calls"] == 2 for row in rows)


def test_launch_overhead_ratios_launch_against_kernels(gpu_tp):
    result = analysis.launch_overhead(gpu_tp)
    assert result["launches"] == 4
    assert result["launch_ms"] == pytest.approx(0.04)  # 4 x 10us
    assert result["device_ms"] == pytest.approx(0.2)  # 4 x 50us
    assert result["launch_per_device_pct"] == pytest.approx(20.0)
    assert result["by_call"][0]["name"] == "cudaLaunchKernel"


def test_a_blocking_sync_is_not_launch_cost(bound_tp):
    """cudaDeviceSynchronize blocks for as long as the kernel it waits on."""
    result = analysis.launch_overhead(bound_tp)
    assert result["launches"] == 5  # not 10, counting the synchronizes
    assert result["launch_ms"] == pytest.approx(0.05)
    assert result["device_ms"] == pytest.approx(5.0)
    assert result["launch_per_device_pct"] == pytest.approx(1.0)
    assert [row["name"] for row in result["by_call"]] == ["cudaLaunchKernel"]


def test_a_blocking_sync_is_reported_on_its_own(bound_tp):
    """CPU waiting on the GPU is worth seeing, just not as launch cost."""
    assert analysis.launch_overhead(bound_tp)["synchronize_ms"] == pytest.approx(5.0)


def test_a_nested_driver_launch_counts_once(tmp_path):
    """cuLaunchKernel nests inside the cudaLaunchKernel that made it."""
    path = write_trace(
        tmp_path / "nested.trace.json.gz",
        [
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "pid": 1,
                "tid": 1,
                "ts": 0,
                "dur": 100,
            },
            {
                "ph": "X",
                "cat": "cuda_driver",
                "name": "cuLaunchKernel",
                "pid": 1,
                "tid": 1,
                "ts": 10,
                "dur": 80,
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "sgemm",
                "pid": 1,
                "tid": 7,
                "ts": 200,
                "dur": 500,
            },
        ],
    )
    handle = open_trace(path)
    try:
        result = analysis.launch_overhead(handle)
    finally:
        handle.close()
    assert result["launches"] == 1
    assert result["launch_ms"] == pytest.approx(0.1)  # the outer 100us only


def test_concurrent_streams_are_reported_separately(tmp_path):
    """Two streams busy over the same window; adding them would exceed 100%."""
    events = [
        {
            "ph": "X",
            "cat": "kernel",
            "name": "a",
            "pid": 1,
            "tid": 7,
            "ts": 0,
            "dur": 100,
        },
        {
            "ph": "X",
            "cat": "kernel",
            "name": "b",
            "pid": 1,
            "tid": 8,
            "ts": 0,
            "dur": 100,
        },
        {
            "ph": "M",
            "name": "thread_name",
            "pid": 1,
            "tid": 7,
            "args": {"name": "stream 7"},
        },
        {
            "ph": "M",
            "name": "thread_name",
            "pid": 1,
            "tid": 8,
            "args": {"name": "stream 8"},
        },
    ]
    handle = open_trace(write_trace(tmp_path / "two.trace.json.gz", events))
    try:
        result = analysis.device_utilization(handle)
    finally:
        handle.close()
    assert {track["track"] for track in result["tracks"]} == {"stream 7", "stream 8"}
    assert all(track["busy_pct"] == pytest.approx(100.0) for track in result["tracks"])
    # No summed total: adding concurrent streams would report 200% busy.
    assert "total_device_busy_ms" not in result


# --- the command line ------------------------------------------------------
#
# The skill invokes this script, so the argument surface is part of the
# contract, not an implementation detail.


def run_cli(capsys, *argv):
    analysis.main(list(argv))
    return capsys.readouterr().out


def test_cli_overview(capsys, trace_path):
    out = run_cli(capsys, "overview", str(trace_path))
    assert "duration_ms" in out
    assert "cpu_op" in out


def test_cli_top_ops_is_a_table(capsys, trace_path):
    out = run_cli(capsys, "top-ops", str(trace_path), "--category", "cpu_op")
    assert "aten::addmm" in out
    assert "self_us" in out


def test_cli_limit_is_honoured(capsys, trace_path):
    out = run_cli(capsys, "top-ops", str(trace_path), "--limit", "3")
    # header + rule + 3 rows
    assert len(out.strip().splitlines()) == 5


def test_cli_json_output(capsys, trace_path):
    out = run_cli(capsys, "--json", "overview", str(trace_path))
    assert json.loads(out)["slices"] > 0


def test_cli_json_after_the_subcommand(capsys, trace_path):
    """`... overview trace --json` is where the flag reads naturally."""
    out = run_cli(capsys, "overview", str(trace_path), "--json")
    assert json.loads(out)["slices"] > 0


def test_cli_sql_from_a_flag(capsys, trace_path):
    out = run_cli(
        capsys, "sql", str(trace_path), "--sql", "select count(*) as n from slice"
    )
    assert "n" in out


def test_cli_sql_from_stdin(capsys, monkeypatch, trace_path):
    """Stdin is the quoting-safe path the skill prefers."""
    monkeypatch.setattr("sys.stdin", io.StringIO("select count(*) as n from slice"))
    out = run_cli(capsys, "sql", str(trace_path), "--sql-file", "-")
    assert "n" in out


def test_cli_rejects_a_missing_trace(capsys, tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        analysis.main(["overview", str(tmp_path / "absent.trace.json.gz")])
    assert "no trace at" in str(exit_info.value)


def test_cli_rejects_a_file_that_is_not_a_trace(tmp_path):
    junk = tmp_path / "junk.trace.json.gz"
    junk.write_bytes(b"not a trace")
    with pytest.raises(SystemExit) as exit_info:
        analysis.main(["overview", str(junk)])
    assert "could not read" in str(exit_info.value)


def test_render_table_on_no_rows():
    assert analysis.render_table([]) == "(no rows)"


# --- the eval fixtures -----------------------------------------------------
#
# The evals are only worth having if their traces exist and still separate the
# cases they claim to. Shipping evals that referenced missing files is exactly
# the failure this guards.

EVALS = pathlib.Path(__file__).parent.parent / ".claude/skills/analyze-trace"


def test_every_eval_file_exists():
    evals = json.loads((EVALS / "evals.json").read_text())
    for case in evals:
        for name in case.get("files", []):
            assert (EVALS / name).is_file(), f"{name} is referenced but missing"


def test_non_optional_evals_ship_their_trace():
    """A fresh clone must be able to run everything not marked optional."""
    evals = json.loads((EVALS / "evals.json").read_text())
    for case in evals:
        if case.get("optional"):
            continue
        assert case.get("files"), f"{case['query']!r} depends on the environment"


def test_the_real_fixture_has_genuine_nesting(capsys):
    """Hand-written traces have no nesting; this one is a real capture."""
    handle = open_trace(EVALS / "evals/cpu-real.json")
    try:
        rows = analysis.top_operators(
            handle, limit=analysis.MAX_LIMIT, category="cpu_op"
        )
    finally:
        handle.close()
    by_name = {row["name"]: row for row in rows}
    # aten::addmm really sits inside aten::linear, so the wrapper's self time
    # is a fraction of the work it contains.
    assert by_name["aten::addmm"]["self_us"] > by_name["aten::linear"]["self_us"]
    assert (
        by_name["aten::linear"]["inclusive_us"] > by_name["aten::addmm"]["inclusive_us"]
    )
    assert rows[0]["name"] == "aten::addmm"


def test_every_eval_has_expected_behavior():
    evals = json.loads((EVALS / "evals.json").read_text())
    assert len(evals) >= 3  # the documented minimum
    for case in evals:
        assert case["expected_behavior"], case["query"]


def test_the_gpu_bound_fixture_is_not_launch_bound():
    """1% launch cost against 5ms of kernels, with the CPU blocked on sync."""
    handle = open_trace(EVALS / "evals/gpu-bound.json")
    try:
        result = analysis.launch_overhead(handle)
    finally:
        handle.close()
    assert result["launch_per_device_pct"] == pytest.approx(1.0)
    assert result["synchronize_ms"] == pytest.approx(5.0)


def test_the_launch_bound_fixture_is_launch_bound():
    """Launch cost far exceeding device time, on a mostly idle stream."""
    handle = open_trace(EVALS / "evals/launch-bound.json")
    try:
        launch = analysis.launch_overhead(handle)
        device = analysis.device_utilization(handle, min_gap_us=10)
    finally:
        handle.close()
    assert launch["launch_per_device_pct"] > 100
    assert launch["synchronize_ms"] == 0
    (track,) = device["tracks"]
    assert track["busy_pct"] < 20
    assert device["idle_gaps"]


def test_the_shapes_fixture_carries_input_dims():
    """The args join the third eval asks for has something to find."""
    handle = open_trace(EVALS / "evals/cpu-shapes.json")
    try:
        result = analysis.run_query(
            handle,
            "select dims.int_value as dim, count(*) as calls from slice s "
            "join args dims on dims.arg_set_id = s.arg_set_id "
            "where s.name = 'aten::mm' and dims.key = 'args.Input Dims[0][0]' "
            "group by dim order by dim",
        )
    finally:
        handle.close()
    assert [row["dim"] for row in result["rows"]] == [128, 256]
