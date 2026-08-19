"""`latency`, `throughput` and `startup`: the model in-process, no HTTP.

Nothing here goes over a socket, so nothing here sees queueing — which is the
point. `latency` isolates the decode path, `throughput` asks what the engine
does with everything at once, `startup` measures only getting ready to serve.

All three take ``--num-iters-warmup``: the first pass through a fresh process
compiles, autotunes and captures, and folding that into a measured iteration is
how a compile time gets published as a latency.
"""

from __future__ import annotations

import hashlib
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any

from walnut.bench.errors import BenchError
from walnut.bench.metrics import parse_percentiles, summarize
from walnut.bench.report import (
    report_latency,
    report_startup,
    report_throughput,
    write_record,
)
from walnut.bench.workload import build_workload


@dataclass
class EngineOptions:
    """The knobs that decide what engine gets built, spelled as `walnut serve`
    spells them."""

    model: str
    device: str = "auto"
    dtype: str = "auto"
    cuda_graph: bool = True
    compile: bool = True
    autotune: bool = True
    max_batch_size: int = 1
    max_seq_len: int | None = None

    def load(self) -> Any:
        from walnut.engine import load_model, parse_dtype, resolve_device

        try:
            device = resolve_device(self.device)
            dtype = parse_dtype(self.dtype)
        except (ValueError, RuntimeError) as exc:
            raise BenchError(str(exc)) from exc
        return load_model(
            self.model,
            device=device,
            dtype=dtype,
            cuda_graph=self.cuda_graph,
            compile=self.compile,
            autotune=self.autotune,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
        )

    def as_flags(self) -> dict[str, Any]:
        return {
            "cuda_graph": self.cuda_graph,
            "compile": self.compile,
            "autotune": self.autotune,
        }


def engine_header(engine: Any, opts: EngineOptions) -> dict[str, Any]:
    """What every in-process record says about the engine it drove."""
    import torch

    return {
        "model": opts.model,
        "device": str(engine.device),
        "dtype": str(engine.dtype).removeprefix("torch."),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "flags": opts.as_flags(),
    }


# ===========================================================================
# latency — single stream, decode path only
# ===========================================================================


def one_request(engine: Any, prompt: str, params: Any) -> dict[str, Any]:
    """Run one request straight at the model, timestamping each token.

    No scheduler, no HTTP. Detokenization happens after the clock stops: it is
    a real serving cost, but it grows with sequence length and inside the loop
    reads as per-token drift no engine change explains. `serve` shows it.
    """
    import torch

    from walnut.engine import Message

    input_ids = engine._encode([Message(role="user", content=prompt)])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    stamps, ids = [], []
    for token in engine.model.iter_generate(
        input_ids,
        params,
        cuda_graph=engine.cuda_graph,
        compile=engine.compile,
        autotune=engine.autotune,
    ):
        stamps.append(time.perf_counter())
        ids.append(token)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    if not stamps:
        raise BenchError("the engine generated nothing; check the prompt")
    return {
        "ttft_ms": (stamps[0] - start) * 1e3,
        "e2e_ms": (end - start) * 1e3,
        "itl_ms": [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:], strict=False)],
        "tokens_out": len(ids),
        "text": engine.tokenizer.decode(ids, skip_special_tokens=True),
    }


def run_latency(
    opts: EngineOptions,
    prompt: str,
    max_tokens: int,
    num_iters: int,
    num_iters_warmup: int,
    temperature: float,
    seed: int | None,
    percentiles: str,
    label: str | None,
    out: str | None,
) -> int:
    from walnut.engine import Message
    from walnut.sampler import SamplingParams

    engine = opts.load()
    params = SamplingParams(
        max_new_tokens=max_tokens, temperature=temperature, seed=seed
    )

    # Discarded: the opening iteration compiles, autotunes and captures, which
    # is `startup`'s subject, not this one's.
    for _ in range(num_iters_warmup):
        one_request(engine, prompt, params)
    runs = [one_request(engine, prompt, params) for _ in range(num_iters)]

    # Every iteration, not just the last: nondeterministic generation is
    # exactly what this catches.
    hashes = sorted({hashlib.sha256(r["text"].encode()).hexdigest()[:12] for r in runs})
    counts = sorted({r["tokens_out"] for r in runs})
    tokens_out = counts[-1]

    quantiles = parse_percentiles(percentiles)
    samples = {
        "ttft": [r["ttft_ms"] for r in runs],
        "tpot": [
            (r["e2e_ms"] - r["ttft_ms"]) / max(1, r["tokens_out"] - 1) for r in runs
        ],
        "ntpot": [r["e2e_ms"] / max(1, r["tokens_out"]) for r in runs],
        "itl": [gap for r in runs for gap in r["itl_ms"]],
        "e2el": [r["e2e_ms"] for r in runs],
    }
    # One sample per iteration for everything but ITL, which pools every gap —
    # a max-deviation over that would describe the workload, not the run.
    metrics = {
        k: summarize(v, quantiles, repeats=k != "itl") for k, v in samples.items()
    }

    record = {
        "mode": "latency",
        "label": label,
        **engine_header(engine, opts),
        "prompt": prompt,
        "prompt_tokens": int(
            engine._encode([Message(role="user", content=prompt)]).shape[1]
        ),
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "tokens_out": tokens_out,
        "tokens_out_all": counts,
        "hit_eos": tokens_out < max_tokens,
        "num_iters_warmup": num_iters_warmup,
        "num_iters": num_iters,
        "percentiles": quantiles,
        "metrics": metrics,
        "tok_per_s": 1e3 / metrics["tpot"]["median"],
        "output_sha": hashes[0] if len(hashes) == 1 else "|".join(hashes),
        "output_deterministic": len(hashes) == 1,
        "output_head": runs[-1]["text"][:80],
    }

    report_latency(record)
    write_record(record, out)
    # Non-zero when the run cannot support the claim it is usually taken for:
    # a nondeterministic greedy generation means the output hash proves nothing.
    return 0 if record["output_deterministic"] else 1


# ===========================================================================
# throughput — every request at once, straight at the engine
# ===========================================================================


def drain(engine: Any, prompts: list[str], config: Any) -> tuple[list[Any], list[str]]:
    """Submit every prompt, then read them all back.

    `Engine.stream` queues and returns before a token exists, so the scheduler
    gets the whole workload at once and batches as it likes. Draining one
    stream at a time is safe: the scheduler runs on its own thread and buffers
    each request's tokens, so an unread stream does not stall the batch.
    """
    streams = [engine.stream([_user(prompt)], config) for prompt in prompts]
    errors: list[str] = []
    for stream in streams:
        try:
            for _ in stream:
                pass
        except Exception as exc:  # a failed request is data, not a crash
            errors.append(f"{type(exc).__name__}: {exc}")
    return streams, errors


def _user(content: str) -> Any:
    from walnut.engine import Message

    return Message(role="user", content=content)


def run_throughput(
    opts: EngineOptions,
    dataset: str,
    num_prompts: int,
    prompt: str,
    input_len: int,
    range_ratio: float,
    tokenizer: str | None,
    max_tokens: int,
    num_iters_warmup: int,
    temperature: float,
    top_p: float,
    ignore_eos: bool,
    seed: int,
    label: str | None,
    out: str | None,
) -> int:
    from walnut.engine import GenerationConfig

    engine = opts.load()
    config = GenerationConfig(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        ignore_eos=ignore_eos,
    )
    prompts = build_workload(
        dataset,
        num_prompts,
        prompt,
        input_len,
        range_ratio,
        tokenizer or opts.model,
        random.Random(seed),
    )

    # A full batch, not one request: compilation is per decode bucket, and a
    # single-request warm-up leaves larger buckets to compile mid-measurement.
    for _ in range(num_iters_warmup):
        drain(engine, prompts[: opts.max_batch_size], GenerationConfig(max_tokens=8))

    start = time.perf_counter()
    streams, errors = drain(engine, prompts, config)
    duration = time.perf_counter() - start

    completed = [s for s in streams if s.usage.completion_tokens > 0]
    if not completed:
        raise BenchError(
            "every request produced nothing:\n  " + "\n  ".join(sorted(set(errors))[:5])
        )
    total_input = sum(s.usage.prompt_tokens for s in completed)
    total_output = sum(s.usage.completion_tokens for s in completed)

    record = {
        "mode": "throughput",
        "label": label,
        **engine_header(engine, opts),
        "max_batch_size": opts.max_batch_size,
        "workload": dataset,
        "prompt": prompt if dataset == "fixed" else None,
        "input_len": input_len if dataset == "random" else None,
        "range_ratio": range_ratio if dataset == "random" else None,
        "num_prompts": num_prompts,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "ignore_eos": ignore_eos,
        "num_iters_warmup": num_iters_warmup,
        "completed": len(completed),
        "failed": len(streams) - len(completed),
        "errors": sorted(set(errors))[:5],
        "duration_s": duration,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "median_prompt_tokens": statistics.median(
            [s.usage.prompt_tokens for s in completed]
        ),
        "output_tokens_all": sorted({s.usage.completion_tokens for s in completed}),
        "request_throughput": len(completed) / duration,
        "input_throughput": total_input / duration,
        "output_throughput": total_output / duration,
        "total_token_throughput": (total_input + total_output) / duration,
    }

    report_throughput(record)
    write_record(record, out)
    engine.close()
    return 0 if record["failed"] == 0 else 1


# ===========================================================================
# startup — only the cost of getting ready to serve
# ===========================================================================


def one_startup(opts: EngineOptions, first_request: bool) -> dict[str, float | None]:
    """Build an engine from nothing and time each phase.

    Three phases, paid at different times and fixed by different work: weights
    off disk, `start` compiling and capturing, then whatever the first request
    still has to do.
    """
    from walnut.engine import GenerationConfig

    load_start = time.perf_counter()
    engine = opts.load()
    load_s = time.perf_counter() - load_start

    prepare_start = time.perf_counter()
    engine.start()
    prepare_s = time.perf_counter() - prepare_start

    request_s = None
    if first_request:
        request_start = time.perf_counter()
        engine.generate([_user("Hello.")], GenerationConfig(max_tokens=8))
        request_s = time.perf_counter() - request_start

    engine.close()
    return {
        "load_s": load_s,
        "prepare_s": prepare_s,
        "first_request_s": request_s,
        "total_s": load_s + prepare_s + (request_s or 0.0),
    }


def run_startup(
    opts: EngineOptions,
    num_iters: int,
    num_iters_warmup: int,
    first_request: bool,
    label: str | None,
    out: str | None,
) -> int:
    """Time start-up and nothing else.

    Each iteration builds a whole engine and throws it away, so the warm-up
    iterations absorb the cold compile and what is left is what a restart costs.
    """
    for _ in range(num_iters_warmup):
        one_startup(opts, first_request)
    runs = [one_startup(opts, first_request) for _ in range(num_iters)]

    # Every phase is a repeat of the same measurement, so `cv` is populated and
    # means what it means everywhere else: the floor a change has to clear.
    phases = {}
    for phase in ("load_s", "prepare_s", "first_request_s", "total_s"):
        values = [v for r in runs if (v := r[phase]) is not None]
        if values:
            phases[phase] = summarize(values, [], repeats=True)

    record = {
        "mode": "startup",
        "label": label,
        "model": opts.model,
        "flags": opts.as_flags(),
        "max_batch_size": opts.max_batch_size,
        "num_iters_warmup": num_iters_warmup,
        "num_iters": num_iters,
        "first_request_included": first_request,
        "phases": phases,
        "iterations": runs,
    }

    report_startup(record)
    write_record(record, out)
    return 0
