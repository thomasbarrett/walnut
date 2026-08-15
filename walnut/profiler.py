"""Torch profiler control for the server and the CLI.

As in vLLM and SGLang: `torch.profiler` records a window, which is exported as
a Chrome trace and read at https://ui.perfetto.dev/. Each trace is written next
to a plain-text ``key_averages()`` table, so a run can also be read without a
browser.

Point `DIR_ENV` at a directory to enable the server endpoints::

    WALNUT_TORCH_PROFILER_DIR=./profiles walnut serve Qwen/Qwen3.5-0.8B
    curl -X POST localhost:8000/start_profile
    ...                              # drive some traffic
    curl -X POST localhost:8000/stop_profile

Profiling stays off unless that variable is set; it costs per-op overhead and
buffers for as long as the window is open.

Two kineto constraints shape the API:

- A window must be stopped on the thread that started it. Crossing threads
  segfaults the interpreter, so `stop` refuses instead.
- ``aten::`` ops are recorded only on the starting thread, while GPU kernels
  are recorded from any thread. A server profile therefore shows the CUDA
  timeline without op names; `walnut profile` runs the model on the profiling
  thread and gets both.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity

logger = logging.getLogger("walnut.profiler")

#: Environment variable naming the output directory (and enabling profiling).
DIR_ENV = "WALNUT_TORCH_PROFILER_DIR"

#: Rows of ``key_averages()`` to keep in the text summary.
SUMMARY_ROWS = 40


@dataclass(frozen=True)
class ProfileArtifacts:
    """The files one `TorchProfiler.stop` wrote."""

    #: Chrome trace, gzipped. Open at https://ui.perfetto.dev/.
    trace: Path
    #: ``key_averages()`` table as plain text, or ``None`` if it failed to
    #: build. See `TorchProfiler.stop`.
    summary: Path | None

    def as_dict(self) -> dict[str, str | None]:
        """JSON form, as returned by ``POST /stop_profile``."""
        return {
            "trace": str(self.trace),
            "summary": str(self.summary) if self.summary else None,
        }


class TorchProfiler:
    """A start/stop handle around one `torch.profiler.profile` window.

    One window at a time, guarded by a lock. Each gets a fresh profiler object,
    since a stopped one cannot be restarted.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        record_shapes: bool = True,
        with_stack: bool = True,
        with_flops: bool = False,
    ) -> None:
        self.directory = Path(directory)
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.with_flops = with_flops
        self._lock = threading.Lock()
        self._profile: torch.profiler.profile | None = None
        self._owner: int | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> TorchProfiler | None:
        """Build a server profiler from `DIR_ENV`, or ``None`` when it is unset.

        Stacks are off: worker-thread frames aren't recorded anyway, and
        collecting them inflates the trace and breaks `summarize`.
        """
        directory = (env if env is not None else os.environ).get(DIR_ENV)
        return cls(directory, with_stack=False) if directory else None

    @property
    def running(self) -> bool:
        """Whether a window is currently open."""
        return self._profile is not None

    def start(self) -> None:
        """Open a window. Raises if one is already open."""
        with self._lock:
            if self._profile is not None:
                raise RuntimeError("profiling is already running")
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            profile = torch.profiler.profile(
                activities=activities,
                record_shapes=self.record_shapes,
                with_stack=self.with_stack,
                with_flops=self.with_flops,
            )
            profile.start()
            self._profile = profile
            self._owner = threading.get_ident()
        logger.info("profile_started", extra={"directory": str(self.directory)})

    def stop(self) -> ProfileArtifacts:
        """Close the window and write the trace, and the summary if it builds.

        Raises if no window is open, or if the caller is not the thread that
        opened it.
        """
        with self._lock:
            profile = self._profile
            if profile is None:
                raise RuntimeError("profiling is not running")
            if self._owner != threading.get_ident():
                raise RuntimeError(
                    "profiling must be stopped on the thread that started it "
                    f"(started on {self._owner}, stopping on "
                    f"{threading.get_ident()})"
                )
            self._profile = None
            self._owner = None
            # Kernels are queued, so the tail of the window is still in flight.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            profile.stop()

        self.directory.mkdir(parents=True, exist_ok=True)
        stem = f"walnut-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        # export_chrome_trace gzips when the name ends in .gz.
        trace = self.directory / f"{stem}.trace.json.gz"
        profile.export_chrome_trace(str(trace))

        # Building the table re-parses kineto's results, which can throw on a
        # profile that exported fine. Don't lose the trace to that.
        summary: Path | None = None
        table = self.directory / f"{stem}.summary.txt"
        try:
            table.write_text(summarize(profile))
            summary = table
        except Exception:
            logger.warning("profile_summary_failed", exc_info=True)

        artifacts = ProfileArtifacts(trace, summary)
        logger.info("profile_stopped", extra=artifacts.as_dict())
        return artifacts


def summarize(profile: torch.profiler.profile, row_limit: int = SUMMARY_ROWS) -> str:
    """Render a finished ``profile`` as a ``key_averages()`` table.

    Sorts by device time where there is any; a CPU-only run has none, and that
    column would be all zeros.
    """
    averages = profile.key_averages()
    sort_by = (
        "self_device_time_total"
        if any(event.self_device_time_total for event in averages)
        else "self_cpu_time_total"
    )
    return averages.table(sort_by=sort_by, row_limit=row_limit)
