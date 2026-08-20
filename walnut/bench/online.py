"""`serve` and `sweep`: what clients see, over HTTP, against a running server.

The only benchmarks here that can observe queueing — and the ones that carry
enough else to drown a 3% kernel win. `walnut.bench.offline` judges those.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from walnut.bench.errors import BenchError
from walnut.bench.metrics import METRICS, summarize
from walnut.bench.report import (
    report_frontier,
    report_serve,
    report_sweep,
    serve_warnings,
    shortfall,
    write_record,
)
from walnut.bench.workload import Shape, build_workload, load_tokenizer


@dataclass
class ServeOptions:
    """Everything `serve` and `sweep` need."""

    base_url: str
    model: str | None
    num_prompts: int
    request_rate: float
    max_concurrency: int | None
    shape: Shape
    tokenizer: str | None
    seed: int
    warmups: int
    goodput: dict[str, float]
    timeout: float
    ready_timeout: float
    label: str | None
    out: str | None
    profile: bool = False


@dataclass
class RequestResult:
    """One request's timings, as the client saw them."""

    success: bool = False
    error: str = ""
    #: How long the request waited in this client for a `--max-concurrency`
    #: slot, between the time it was due to be sent and the time it went out.
    queue_wait: float = 0.0
    start: float = 0.0
    ttft: float = 0.0
    latency: float = 0.0
    itl: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    #: Deltas received. Not the same as tokens — see `send_request`.
    deltas: int = 0
    usage_reported: bool = False
    #: The stream ended without its `[DONE]` sentinel, so the response was cut
    #: short. Its latencies are short for the wrong reason and must not be
    #: averaged in with the rest.
    truncated: bool = False
    text: str = ""

    @property
    def tpot(self) -> float:
        if self.output_tokens < 2:
            return 0.0
        return (self.latency - self.ttft) / (self.output_tokens - 1)


def request_metrics(result: RequestResult) -> dict[str, float]:
    """One request's value for every metric, in seconds.

    ITL reduces to the worst gap: averaging one two-second stall against
    ninety-nine fast gaps is how a stall gets reported as healthy.
    """
    return {
        "ttft": result.ttft,
        "tpot": result.tpot,
        "itl": max(result.itl) if result.itl else 0.0,
        "e2el": result.latency,
    }


def meets_slos(result: RequestResult, slos: dict[str, float]) -> bool:
    """Whether one request cleared every SLO it was held to.

    Every one, not any: a request that answered in 80 ms and then stalled for
    two seconds served nobody, and each metric alone scores it a success.
    """
    measured = request_metrics(result)
    return all(measured[key] <= limit for key, limit in slos.items())


def peak_concurrency(results: list[RequestResult]) -> int:
    """Most requests in flight at the server at once.

    With ``--max-concurrency`` set it cannot exceed the limit and says nothing;
    read ``queue_wait`` instead. Without one, a peak above the server's
    ``--max-batch-size`` means requests were queueing inside the engine.
    """
    events = []
    for result in results:
        events.append((result.start, 1))
        events.append((result.start + result.latency, -1))
    events.sort()
    live = peak = 0
    for _, delta in events:
        live += delta
        peak = max(peak, live)
    return peak


async def send_request(
    client: Any, url: str, payload: dict[str, Any], timeout: float
) -> RequestResult:
    """Stream one chat completion, timestamping every content delta.

    Timings include detokenization and the HTTP hop by design: this measures
    what a client experiences. Token counts come from the usage chunk, never
    from counting deltas — one delta can carry two tokens, which would inflate
    TPOT silently and only on multi-byte output.
    """
    result = RequestResult(start=time.perf_counter())
    previous = result.start
    pieces: list[str] = []
    done = False
    try:
        async with client.stream(
            "POST", url, json=payload, timeout=timeout
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                result.error = f"HTTP {response.status_code}: {body[:200]!r}"
                return result
            async for raw in response.aiter_lines():
                if not raw.startswith("data: "):
                    continue
                data = raw[len("data: ") :]
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if usage := chunk.get("usage"):
                    result.prompt_tokens = usage.get("prompt_tokens", 0)
                    result.output_tokens = usage.get("completion_tokens", 0)
                    result.usage_reported = True
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                content = choices[0].get("delta", {}).get("content")
                if not content:
                    # The opening role delta carries no text; it is the
                    # response's first byte, not its first token.
                    continue
                now = time.perf_counter()
                if not result.deltas:
                    result.ttft = now - result.start
                else:
                    result.itl.append(now - previous)
                previous = now
                result.deltas += 1
                pieces.append(content)
        result.latency = previous - result.start
        result.text = "".join(pieces)
        if not result.usage_reported:
            result.output_tokens = result.deltas
        # A stream that stopped without its sentinel was cut short — the server
        # raised mid-response. It has a small latency and a small token count,
        # and counting it as a success drags every reported percentile down.
        result.truncated = not done
        result.success = result.deltas > 0 and done
        if not result.success and not result.error:
            result.error = (
                "the stream ended without [DONE] after "
                f"{result.deltas} deltas (truncated mid-response)"
                if result.deltas
                else "the server streamed no content"
            )
    except Exception as exc:  # a failed request is data, not a crash
        result.error = f"{type(exc).__name__}: {exc}"
    return result


async def wait_until_ready(client: Any, base_url: str, timeout: float) -> list[str]:
    """Block until the server answers /models, returning the ids it serves.

    Without this the first requests meet a server still loading weights, and
    the run reports a load time as a latency.
    """
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            response = await client.get(f"{base_url}/models", timeout=5.0)
            if response.status_code == 200:
                return [m["id"] for m in response.json().get("data", [])]
            last = f"HTTP {response.status_code}"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(0.5)
    raise BenchError(f"server at {base_url} not ready after {timeout:g}s ({last})")


async def drive(
    client: Any,
    url: str,
    prompts: list[str],
    payload: dict[str, Any],
    rate: float,
    max_concurrency: int | None,
    timeout: float,
    rng: random.Random,
) -> tuple[list[RequestResult], float, float, float]:
    """Submit every prompt on its arrival schedule; collect what came back.

    Returns the results, the run's wall time, the span requests were
    *submitted* over, and the span the schedule called for. Throughput over the
    whole run is depressed by the drain tail; submitted-against-scheduled is
    what says whether this client kept up.

    Arrival times are absolute, accumulated from the start. Sleeping per
    iteration restarts the clock after every wake-up, so overhead accumulates
    and the client silently offers less load than configured.

    ``max_concurrency`` gates execution, not submission. The wait for a slot is
    recorded as ``queue_wait``, not folded into TTFT: it happened here, not in
    the engine.
    """
    from walnut.bench.workload import arrival_delays

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def one(prompt: str) -> RequestResult:
        arrived = time.perf_counter()
        body = {**payload, "messages": [{"role": "user", "content": prompt}]}
        if semaphore is None:
            result = await send_request(client, url, body, timeout)
        else:
            async with semaphore:
                result = await send_request(client, url, body, timeout)
        result.queue_wait = result.start - arrived
        return result

    schedule = list(itertools.accumulate(arrival_delays(len(prompts), rate, rng)))
    tasks = []
    start = time.perf_counter()
    for prompt, at in zip(prompts, schedule, strict=True):
        now = time.perf_counter() - start
        if at > now:
            await asyncio.sleep(at - now)
        tasks.append(asyncio.create_task(one(prompt)))
    submit_span = time.perf_counter() - start
    results = await asyncio.gather(*tasks)
    return list(results), time.perf_counter() - start, submit_span, schedule[-1]


def serve_payload(opts: ServeOptions, model: str) -> dict[str, Any]:
    """Greedy, and every request held to exactly ``max_tokens``.

    Neither is a flag: sampling cost is only legible with the scheduler and HTTP
    out of the way (`walnut bench latency`), and ragged output lengths give
    latencies this harness already refuses to compare.
    """
    return {
        "model": model,
        "max_tokens": opts.shape.output_len,
        "temperature": 0.0,
        "seed": opts.seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }


async def warm_up(
    client: Any, url: str, payload: dict[str, Any], prompt: str, opts: ServeOptions
) -> None:
    """Discarded, and sequential: the first request through a cold server pays
    for compilation and graph capture. `walnut bench startup` measures that."""
    for _ in range(opts.warmups):
        warm = await send_request(
            client,
            url,
            {**payload, "messages": [{"role": "user", "content": prompt}]},
            opts.timeout,
        )
        if not warm.success:
            raise BenchError(f"warm-up request failed: {warm.error}")


def build_serve_record(
    opts: ServeOptions,
    model: str,
    results: list[RequestResult],
    duration: float,
    submit_span: float,
    scheduled_span: float,
    rate: float,
) -> dict[str, Any]:
    ok = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    if not ok:
        errors = sorted({r.error for r in failed})[:5]
        raise BenchError("every request failed:\n  " + "\n  ".join(errors))

    slos = opts.goodput
    good = sum(1 for r in ok if meets_slos(r, slos)) if slos else None
    total_output = sum(r.output_tokens for r in ok)
    total_input = sum(r.prompt_tokens for r in ok)

    # Per-request for every metric but ITL, which is pooled per gap: a
    # distribution over tokens is what shows a stall, and one value per request
    # would average it away before it was ever plotted.
    per_request = [request_metrics(r) for r in ok]
    samples: dict[str, list[float]] = {
        "itl": [gap * 1e3 for r in ok for gap in r.itl],
    }
    for key, *_ in METRICS:
        if key == "itl":
            continue
        # TPOT is undefined for a response too short to have a decode phase;
        # including a zero there drags the median down.
        floor = 2 if key == "tpot" else 1
        samples[key] = [
            m[key] * 1e3
            for m, r in zip(per_request, ok, strict=True)
            if r.output_tokens >= floor
        ]

    return {
        "mode": "serve",
        "label": opts.label,
        "model": model,
        "base_url": opts.base_url,
        # Workload and traffic shape, kept apart: a latency read against a run
        # that changed either is the easiest fake result in serving. The shape
        # travels by name so two records can be checked for comparability
        # rather than assumed to be.
        "shape": opts.shape.name,
        "input_len": opts.shape.input_len,
        "jitter": opts.shape.jitter,
        "num_prompts": opts.num_prompts,
        "request_rate": rate,
        "max_concurrency": opts.max_concurrency,
        "max_tokens": opts.shape.output_len,
        "seed": opts.seed,
        "warmups": opts.warmups,
        "completed": len(ok),
        "failed": len(failed),
        "errors": sorted({r.error for r in failed})[:5],
        "duration_s": duration,
        # Three clocks. `duration_s` spans submission plus the drain tail, so
        # `request_throughput` sits below the offered rate by construction.
        # `achieved_rate` is over submission alone and is the only one that may
        # be read against `--request-rate`. `scheduled_span_s` is what the
        # arrival process asked for; see `shortfall`.
        "submit_span_s": submit_span,
        "scheduled_span_s": scheduled_span,
        "achieved_rate": (
            (len(results) - 1) / submit_span if submit_span > 0 else float("inf")
        ),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        # Median rather than mean: with `--dataset random` the lengths are a
        # spread, and the median says what a typical request carried.
        "median_prompt_tokens": statistics.median([r.prompt_tokens for r in ok]),
        "output_tokens_all": sorted({r.output_tokens for r in ok}),
        # Exact counts require the server's usage chunk. Without it the tool
        # counts deltas, which runs low on multi-byte output and inflates TPOT.
        "token_counts_exact": all(r.usage_reported for r in ok),
        "request_throughput": len(ok) / duration,
        "input_throughput": total_input / duration,
        "output_throughput": total_output / duration,
        "total_token_throughput": (total_input + total_output) / duration,
        # Little's law: mean requests resident over the run. Peak is the
        # diagnostic — above --max-batch-size means the engine queued
        # internally, which the mean smooths away.
        "concurrency": sum(r.latency for r in ok) / duration,
        "peak_concurrency": peak_concurrency(ok),
        "queue_wait_ms": summarize([r.queue_wait * 1e3 for r in ok]),
        "truncated": sum(1 for r in results if r.truncated),
        "goodput_slos": {k: v * 1e3 for k, v in slos.items()} or None,
        # In seconds too: what `at_slo` reads back, undivided.
        "goodput_slos_seconds": dict(slos),
        "goodput": (good / duration) if good is not None else None,
        "goodput_fraction": (good / len(ok)) if good is not None else None,
        "metrics": {k: summarize(v) for k, v in samples.items()},
        "output_head": ok[-1].text[:80],
    }


async def serve_once(
    client: Any,
    opts: ServeOptions,
    model: str,
    prompts: list[str],
    payload: dict[str, Any],
    url: str,
    rate: float,
    rng: random.Random,
) -> dict[str, Any]:
    """One measured run at one offered rate. `sweep` calls this per rung."""
    print(
        f"{opts.shape.name}: {len(prompts)} prompts at "
        f"{'unlimited' if rate == float('inf') else rate}"
        f" req/s (Poisson), "
        f"max concurrency {opts.max_concurrency or 'unlimited'}",
        file=sys.stderr,
    )
    results, duration, submit_span, scheduled_span = await drive(
        client,
        url,
        prompts,
        payload,
        rate,
        opts.max_concurrency,
        opts.timeout,
        rng,
    )
    return build_serve_record(
        opts, model, results, duration, submit_span, scheduled_span, rate
    )


async def _prepare(client: Any, opts: ServeOptions) -> tuple[str, list[str], str, dict]:
    """Resolve the model, build the workload, and warm the server once."""
    base_url = opts.base_url.rstrip("/")
    url = f"{base_url}/chat/completions"
    served = await wait_until_ready(client, base_url, opts.ready_timeout)
    model = opts.model or (served[0] if served else None)
    if model is None:
        raise BenchError(f"{base_url}/models advertises no model")

    rng = random.Random(opts.seed)
    tokenizer = (
        load_tokenizer(opts.tokenizer or opts.model or model)
        if opts.shape.input_len is not None
        else None
    )
    prompts = build_workload(opts.shape, opts.num_prompts, tokenizer, rng)
    payload = serve_payload(opts, model)
    await warm_up(client, url, payload, prompts[0], opts)
    return model, prompts, url, payload


def _client() -> Any:
    import httpx

    return httpx.AsyncClient(
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None)
    )


async def run_serve(opts: ServeOptions) -> int:
    rng = random.Random(opts.seed)
    base_url = opts.base_url.rstrip("/")

    async with _client() as client:
        model, prompts, url, payload = await _prepare(client, opts)

        if opts.profile:
            await client.post(f"{base_url.removesuffix('/v1')}/start_profile", json={})
            print("profiling window opened", file=sys.stderr)
        try:
            record = await serve_once(
                client, opts, model, prompts, payload, url, opts.request_rate, rng
            )
        finally:
            # Always, even on Ctrl-C: an open profiling window keeps filling
            # the server's trace buffer until it dies.
            if opts.profile:
                stopped = await client.post(
                    f"{base_url.removesuffix('/v1')}/stop_profile", timeout=600.0
                )
                print(f"trace: {stopped.text}", file=sys.stderr)

    report_serve(record)
    write_record(record, opts.out)
    return 0 if record["failed"] == 0 else 1


async def run_sweep(
    opts: ServeOptions, ladder: list[float], goodput_floor: float
) -> int:
    """Open loop, up a ladder of offered rates, stopping at the knee."""
    rng = random.Random(opts.seed)
    rungs: list[dict[str, Any]] = []
    stopped = ""

    async with _client() as client:
        # Warmed once, not per rung: the second rung is not cold.
        model, prompts, url, payload = await _prepare(client, opts)

        for rate in ladder:
            record = await serve_once(
                client, opts, model, prompts, payload, url, rate, rng
            )
            # Per rung, not once at the end: a server that quietly ignored
            # --ignore-eos invalidates every rung, and the sweep should say so
            # on the one where it was first visible rather than after the ladder.
            serve_warnings(record)
            rungs.append(record)
            # Stop at the first rung the server could not absorb. Rungs above
            # it measure a queue, not an engine, and running them costs GPU
            # time to produce numbers that must not be quoted.
            if shortfall(record) > 0.02:
                stopped = (
                    f"stopped at {rate:g} req/s: the client fell "
                    f"{shortfall(record) * 100:.0f}% behind its arrival "
                    "schedule, so it can no longer offer this load. Nothing "
                    "above this rate would describe the engine."
                )
                break
            fraction = record["goodput_fraction"]
            if fraction is not None and fraction < goodput_floor:
                stopped = (
                    f"stopped at {rate:g} req/s: goodput fell to "
                    f"{fraction * 100:.0f}%, below the {goodput_floor * 100:.0f}% "
                    "floor. This is the knee."
                )
                break

    report_sweep(rungs, stopped, opts.goodput)
    write_record(
        {
            "mode": "sweep",
            # Named for the flag that carried the ladder, so a record says
            # which knob was turned rather than which word the report used.
            "axis": "request_rate",
            "label": opts.label,
            "model": rungs[0]["model"],
            "shape": opts.shape.name,
            "ladder": ladder,
            "goodput_floor": goodput_floor,
            "stopped": stopped,
            "rungs": rungs,
        },
        opts.out,
    )
    return 0


async def run_frontier(opts: ServeOptions, ladder: list[int]) -> int:
    """Closed loop, up a ladder of concurrency limits.

    `run_sweep` offers traffic and finds the rate the engine stops absorbing;
    this holds a fixed number of requests in flight and asks what throughput
    that buys at what per-stream cost. Under open-loop arrivals concurrency is
    an outcome and not a setting, so that axis does not exist there.

    Every rung runs. A closed loop cannot build an unbounded queue, so no rung
    invalidates the ones above it — each is an operating point somebody might
    choose, and the curve is the answer.
    """
    rng = random.Random(opts.seed)
    rungs: list[dict[str, Any]] = []

    async with _client() as client:
        model, prompts, url, payload = await _prepare(client, opts)
        for limit in ladder:
            rung = dataclasses.replace(opts, max_concurrency=limit)
            record = await serve_once(
                client, rung, model, prompts, payload, url, float("inf"), rng
            )
            serve_warnings(record)
            rungs.append(record)

    report_frontier(rungs, opts.goodput)
    write_record(
        {
            "mode": "sweep",
            "axis": "max_concurrency",
            "label": opts.label,
            "model": rungs[0]["model"],
            "shape": opts.shape.name,
            "ladder": ladder,
            "rungs": rungs,
        },
        opts.out,
    )
    return 0
