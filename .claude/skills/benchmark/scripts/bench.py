"""Single-stream latency benchmark for walnut, and a comparator for two runs.

Reports the standard LLM serving metrics, measured client-side off the engine's
streaming interface, the way a user experiences them:

    TTFT  time to first token   -- prefill + CUDA graph capture + sampling
    ITL   inter-token latency   -- one sample per token after the first
    TPOT  (e2e - TTFT) / (tokens_out - 1)
    e2e   end-to-end request latency

walnut serves one request at a time, so there is no request rate, concurrency
or goodput to sweep: these are single-stream numbers, and `tok/s` is per-stream
output throughput (1000 / TPOT), not system throughput.

    BENCH=.claude/skills/benchmark/scripts/bench.py
    uv run python $BENCH run MODEL --label baseline -o before.json
    uv run python $BENCH compare before.json after.json

Sampling is greedy and the output is hashed into the record, so `compare` says
plainly when a change altered what the model produces -- the first thing to
check before reading any speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROMPT = "Explain how a transformer works."


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, so small samples stay honest."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(q / 100 * len(ordered) + 0.5) - 1))
    return ordered[index]


def _one_request(engine: Any, prompt: str, tokens: int) -> dict[str, Any]:
    """Stream one greedy request, timestamping every chunk."""
    import torch

    from walnut.engine import GenerationConfig, Message

    messages = [Message(role="user", content=prompt)]
    config = GenerationConfig(max_tokens=tokens, temperature=0.0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    stamps, text = [], ""
    for chunk in engine.stream(messages, config):
        stamps.append(time.perf_counter())
        text += chunk
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    if not stamps:
        raise RuntimeError("the engine streamed nothing; check the prompt")
    return {
        "ttft_ms": (stamps[0] - start) * 1e3,
        "e2e_ms": (end - start) * 1e3,
        # One sample per token boundary. A chunk is a token here: `stream`
        # yields per decoded token, withholding only partial UTF-8 sequences.
        "itl_ms": [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:], strict=False)],
        "text": text,
    }


def run(args: argparse.Namespace) -> int:
    import torch

    from walnut.engine import GenerationConfig, Message, load_model, parse_dtype
    from walnut.engine import resolve_device as resolve

    engine = load_model(
        args.model,
        device=resolve(args.device),
        dtype=parse_dtype(args.dtype),
        cuda_graph=args.cuda_graph,
        compile=args.compile,
    )

    # Cold: compilation, autotuning and lazy init. A real cost, paid once per
    # process, and the only run in which it is visible.
    cold = time.perf_counter()
    engine.generate(
        [Message(role="user", content=args.prompt)],
        GenerationConfig(max_tokens=8, temperature=0.0),
    )
    first_request_ms = (time.perf_counter() - cold) * 1e3

    for _ in range(args.warmup):
        _one_request(engine, args.prompt, args.tokens)
    runs = [_one_request(engine, args.prompt, args.tokens) for _ in range(args.repeats)]

    text = runs[-1]["text"]
    tokens_out = len(engine.tokenizer.encode(text))
    short = tokens_out < args.tokens
    if short:
        print(
            f"! generation stopped at {tokens_out} of {args.tokens} tokens (EOS). "
            "Per-token metrics are still valid; e2e is not comparable against a "
            "run that generated a different number of tokens.",
            file=sys.stderr,
        )

    itl = [sample for r in runs for sample in r["itl_ms"]]
    ttft = [r["ttft_ms"] for r in runs]
    e2e = [r["e2e_ms"] for r in runs]
    tpot = [(r["e2e_ms"] - r["ttft_ms"]) / max(1, len(r["itl_ms"])) for r in runs]

    record = {
        "label": args.label,
        "model": args.model,
        "device": str(engine.device),
        "dtype": str(engine.dtype).removeprefix("torch."),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "host": platform.node(),
        "torch": torch.__version__,
        "flags": {"cuda_graph": args.cuda_graph, "compile": args.compile},
        "prompt": args.prompt,
        "prompt_tokens": len(engine.tokenizer.encode(args.prompt)),
        "max_tokens": args.tokens,
        "tokens_out": tokens_out,
        "hit_eos": short,
        "warmup": args.warmup,
        "repeats": args.repeats,
        # Medians for the per-request metrics, percentiles for the per-token one.
        "ttft_ms": statistics.median(ttft),
        "e2e_ms": statistics.median(e2e),
        "tpot_ms": statistics.median(tpot),
        "tok_per_s": 1e3 / statistics.median(tpot),
        "itl_mean_ms": statistics.fmean(itl),
        "itl_p50_ms": _percentile(itl, 50),
        "itl_p90_ms": _percentile(itl, 90),
        "itl_p99_ms": _percentile(itl, 99),
        "itl_max_ms": max(itl),
        "first_request_ms": first_request_ms,
        "ttft_all_ms": ttft,
        "e2e_all_ms": e2e,
        # Greedy: a change in this hash is a change in behavior, not noise.
        "output_sha": hashlib.sha256(text.encode()).hexdigest()[:12],
        "output_head": text[:80],
    }

    print(json.dumps(record, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


def _change(before: float, after: float) -> str:
    if not before:
        return "n/a"
    return f"{(after - before) / before * 100:+.1f}%"


ROWS = [
    ("TTFT (ms)", "ttft_ms"),
    ("TPOT (ms/token)", "tpot_ms"),
    ("output tok/s", "tok_per_s"),
    ("ITL mean (ms)", "itl_mean_ms"),
    ("ITL p99 (ms)", "itl_p99_ms"),
    ("e2e (ms)", "e2e_ms"),
    ("first request (ms)", "first_request_ms"),
]


def compare(args: argparse.Namespace) -> int:
    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())

    # A benchmark comparison is only as good as the two runs matching.
    for field in ("model", "device", "gpu", "prompt", "max_tokens", "dtype"):
        if before.get(field) != after.get(field):
            print(
                f"! {field} differs: {before.get(field)!r} vs {after.get(field)!r} "
                "-- these runs are not comparable",
                file=sys.stderr,
            )

    print(f"before: {before.get('label') or args.before}  {before['flags']}")
    print(f"after:  {after.get('label') or args.after}  {after['flags']}")
    print(
        f"config: {before['model']} on {before['gpu'] or before['device']}, "
        f"{before['dtype']}, {before['prompt_tokens']} prompt tokens -> "
        f"{before['tokens_out']} output tokens, "
        f"median of {before['repeats']} runs\n"
    )

    print(f"{'':22} {'before':>10} {'after':>10} {'change':>10}")
    for name, key in ROWS:
        b, a = before.get(key), after.get(key)
        if b is None or a is None:
            continue
        print(f"{name:22} {b:10.2f} {a:10.2f} {_change(b, a):>10}")

    print()
    if before["output_sha"] == after["output_sha"]:
        print(f"output: identical ({before['output_sha']})")
    else:
        print(f"output: CHANGED {before['output_sha']} -> {after['output_sha']}")
        print(f"  before: {before['output_head']!r}")
        print(f"  after:  {after['output_head']!r}")
        print("  Explain this before claiming a speedup.")
    if before["tokens_out"] != after["tokens_out"]:
        print(
            f"! token counts differ ({before['tokens_out']} vs "
            f"{after['tokens_out']}); e2e is not comparable"
        )

    print(f"\nspeedup: {before['tpot_ms'] / after['tpot_ms']:.2f}x on TPOT")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="benchmark one configuration")
    r.add_argument("model")
    r.add_argument("--prompt", default=PROMPT)
    r.add_argument("--tokens", type=int, default=128, help="max output tokens")
    r.add_argument("--repeats", type=int, default=5, help="measured requests")
    r.add_argument("--warmup", type=int, default=1, help="discarded requests")
    r.add_argument("--device", default="auto")
    r.add_argument("--dtype", default="auto")
    r.add_argument("--cuda-graph", action="store_true", default=True)
    r.add_argument("--no-cuda-graph", dest="cuda_graph", action="store_false")
    r.add_argument("--compile", action="store_true", default=True)
    r.add_argument("--no-compile", dest="compile", action="store_false")
    r.add_argument("--label", default=None, help="name for this run in `compare`")
    r.add_argument("-o", "--out", default=None, help="write the JSON record here")
    r.set_defaults(func=run)

    c = sub.add_parser("compare", help="diff two records from `run`")
    c.add_argument("before")
    c.add_argument("after")
    c.set_defaults(func=compare)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
