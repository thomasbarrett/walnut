"""Single-stream latency benchmark for walnut, and a comparator for two runs.

Reports the standard serving metrics:

    TTFT  time to first token   -- prefill + CUDA graph capture + first sample
    ITL   inter-token latency   -- one sample per token after the first
    TPOT  (e2e - TTFT) / (tokens_out - 1)
    e2e   end-to-end request latency

Timestamps come from the engine's own token stream (`iter_generate`), one per
token, and the text is decoded once at the end. Timing `Engine.stream` instead
would be wrong twice: it yields per *decoded chunk*, so a multi-token character
(emoji, CJK) silently halves the sample count, and it re-detokenizes the whole
sequence every step, putting an O(n^2) Python cost inside the measured gap.

walnut serves one request at a time, so there is no request rate, concurrency
or goodput to sweep: these are single-stream numbers, and `tok/s` is per-stream
output throughput (1000 / TPOT), not system throughput.

    BENCH=.claude/skills/benchmark/scripts/bench.py
    uv run python $BENCH run MODEL --label baseline -o before.json
    uv run python $BENCH compare before.json after.json

`compare` exits non-zero and refuses to print a speedup when the two runs are
not comparable, so a mismatch cannot be pasted into a PR as a result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROMPT = "Explain how a transformer works."

# Fields that must match for two records to describe the same experiment.
# `flags` and `temperature` are here because flipping one is the easiest way to
# manufacture a speedup without writing any code.
COMPARABLE = (
    "model",
    "device",
    "gpu",
    "dtype",
    "prompt",
    "max_tokens",
    "temperature",
    "flags",
    "torch",
)


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value at or above rank q."""
    ordered = sorted(values)
    rank = math.ceil(q / 100 * len(ordered))
    return ordered[min(len(ordered) - 1, max(0, rank - 1))]


def _git_state() -> dict[str, Any]:
    """Commit and dirtiness, so two records can be traced back to code."""

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"git_sha": head, "git_dirty": bool(status) if status is not None else None}


def _inductor_cache_entries() -> int | None:
    """Files in Inductor's on-disk cache, or None if torch won't say where.

    Counted either side of the first request: the cache is keyed on the graph,
    so growth means this process compiled and a hit leaves it alone.
    """
    try:
        from torch._inductor.runtime.cache_dir_utils import cache_dir
    except ImportError:
        try:
            from torch._inductor.codecache import cache_dir  # type: ignore[no-redef]
        except ImportError:
            return None
    root = Path(cache_dir())
    if not root.is_dir():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file())


def _one_request(engine: Any, prompt: str, params: Any) -> dict[str, Any]:
    """Run one request, timestamping each generated token.

    Detokenization happens after the clock stops: it is a real serving cost but
    it grows with sequence length, and inside the loop it would show up as
    per-token drift that no engine change can explain.
    """
    import torch

    from walnut.engine import Message

    input_ids = engine._encode([Message(role="user", content=prompt)])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    stamps, ids = [], []
    for token in engine.model.iter_generate(
        input_ids, params, cuda_graph=engine.cuda_graph, compile=engine.compile
    ):
        stamps.append(time.perf_counter())
        ids.append(token)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    if not stamps:
        raise RuntimeError("the engine generated nothing; check the prompt")
    return {
        "ttft_ms": (stamps[0] - start) * 1e3,
        "e2e_ms": (end - start) * 1e3,
        "itl_ms": [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:], strict=False)],
        "tokens_out": len(ids),
        "text": engine.tokenizer.decode(ids, skip_special_tokens=True),
    }


def run(args: argparse.Namespace) -> int:
    import torch

    from walnut.engine import GenerationConfig, Message, load_model, parse_dtype
    from walnut.engine import resolve_device as resolve
    from walnut.sampler import SamplingParams

    engine = load_model(
        args.model,
        device=resolve(args.device),
        dtype=parse_dtype(args.dtype),
        cuda_graph=args.cuda_graph,
        compile=args.compile,
    )

    # Cold: compilation, autotuning and lazy init. Weight loading already
    # happened in load_model, so it is not in this number.
    cache_before = _inductor_cache_entries()
    cold = time.perf_counter()
    engine.generate(
        [Message(role="user", content=args.prompt)],
        GenerationConfig(max_tokens=8, temperature=args.temperature),
    )
    first_request_ms = (time.perf_counter() - cold) * 1e3
    cache_after = _inductor_cache_entries()

    compiled = (
        None
        if cache_before is None or cache_after is None
        else cache_after > cache_before
    )
    if compiled:
        print(
            "! the first request compiled from a cold Inductor cache, so "
            "`first request (ms)` is a compile time. Re-run in a fresh process "
            "for a number comparable with a warm baseline.",
            file=sys.stderr,
        )

    params = SamplingParams(
        max_new_tokens=args.tokens, temperature=args.temperature, seed=args.seed
    )
    for _ in range(args.warmup):
        _one_request(engine, args.prompt, params)
    runs = [_one_request(engine, args.prompt, params) for _ in range(args.repeats)]

    # Every run is hashed, not just the last: a change that makes generation
    # nondeterministic is exactly what this is meant to catch.
    hashes = sorted({hashlib.sha256(r["text"].encode()).hexdigest()[:12] for r in runs})
    if len(hashes) > 1:
        print(
            f"! generation is not deterministic across repeats: {hashes}. "
            "Per-token metrics are still valid; the output hash is not.",
            file=sys.stderr,
        )

    counts = sorted({r["tokens_out"] for r in runs})
    tokens_out = counts[-1]
    if tokens_out < args.tokens:
        print(
            f"! generation stopped at {tokens_out} of {args.tokens} tokens (EOS). "
            "Per-token metrics are still valid; e2e is not comparable against a "
            "run that generated a different number of tokens.",
            file=sys.stderr,
        )

    itl = [sample for r in runs for sample in r["itl_ms"]]
    ttft = [r["ttft_ms"] for r in runs]
    e2e = [r["e2e_ms"] for r in runs]
    tpot = [(r["e2e_ms"] - r["ttft_ms"]) / max(1, r["tokens_out"] - 1) for r in runs]

    record = {
        "label": args.label,
        "model": args.model,
        "device": str(engine.device),
        "dtype": str(engine.dtype).removeprefix("torch."),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "host": platform.node(),
        "torch": torch.__version__,
        **_git_state(),
        "flags": {"cuda_graph": args.cuda_graph, "compile": args.compile},
        "prompt": args.prompt,
        "prompt_tokens": int(
            engine._encode([Message(role="user", content=args.prompt)]).shape[1]
        ),
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": args.tokens,
        "tokens_out": tokens_out,
        "tokens_out_all": counts,
        "hit_eos": tokens_out < args.tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
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
        # None if torch won't name its cache dir; `compare` then stays quiet.
        "first_request_compiled": compiled,
        # Per-run values, so dispersion can be checked downstream.
        "ttft_all_ms": ttft,
        "e2e_all_ms": e2e,
        "tpot_all_ms": tpot,
        "output_sha": hashes[0] if len(hashes) == 1 else "|".join(hashes),
        "output_deterministic": len(hashes) == 1,
        "output_head": runs[-1]["text"][:80],
        "output_text": runs[-1]["text"],
    }

    print(json.dumps({k: v for k, v in record.items() if k != "output_text"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


def _change(before: float, after: float) -> str:
    if not before:
        return "n/a"
    return f"{(after - before) / before * 100:+.1f}%"


def _spread(values: list[float] | None) -> str:
    """Widest deviation from the median, as a percentage of it.

    Reads directly against the `change` column: a +1.8% move next to ±2.4%
    dispersion is not a result.
    """
    if not values or len(values) < 2:
        return ""
    mid = statistics.median(values)
    if not mid:
        return ""
    return f"±{max(abs(v - mid) for v in values) / mid * 100:.1f}%"


def _cell(value: float, values: list[float] | None) -> str:
    """A metric and its dispersion together, so neither is read without the other."""
    spread = _spread(values)
    return f"{value:.2f} {spread}" if spread else f"{value:.2f}"


ROWS = [
    ("TTFT (ms)", "ttft_ms", "ttft_all_ms"),
    ("TPOT (ms/token)", "tpot_ms", "tpot_all_ms"),
    ("output tok/s", "tok_per_s", None),
    ("ITL mean (ms)", "itl_mean_ms", None),
    ("ITL p99 (ms)", "itl_p99_ms", None),
    ("ITL max (ms)", "itl_max_ms", None),
    ("e2e (ms)", "e2e_ms", "e2e_all_ms"),
    ("first request (ms)", "first_request_ms", None),
]


def _first_difference(a: str, b: str) -> str:
    """Where two outputs diverge — divergence is usually far from the start."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            lo = max(0, i - 20)
            return f"at char {i}: {a[lo : i + 20]!r} vs {b[lo : i + 20]!r}"
    return f"one is a prefix of the other ({len(a)} vs {len(b)} chars)"


def compare(args: argparse.Namespace) -> int:
    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())

    mismatched = [f for f in COMPARABLE if before.get(f) != after.get(f)]
    print(f"before: {before.get('label') or args.before}  {before['flags']}")
    print(f"after:  {after.get('label') or args.after}  {after['flags']}")

    if mismatched:
        # On stdout, and fatal: a mismatch pasted into a PR is worse than no
        # number at all, and stderr does not survive copy-paste.
        print("\nNOT COMPARABLE — these runs differ in what they measured:")
        for field in mismatched:
            print(f"  {field}: {before.get(field)!r} -> {after.get(field)!r}")
        print("\nRe-run both sides with the same configuration.")
        return 1

    print(
        f"config: {before['model']} on {before['gpu'] or before['device']}, "
        f"{before['dtype']}, temperature {before['temperature']}, "
        f"{before['prompt_tokens']} prompt tokens -> {before['tokens_out']} output "
        f"tokens, median of {before['repeats']} runs"
    )
    for name, rec in (("before", before), ("after", after)):
        sha, dirty = rec.get("git_sha"), rec.get("git_dirty")
        mark = "+dirty" if dirty else ""
        print(f"  {name}: {(sha or 'unknown')[:12]}{mark}")

    print(f"\n{'':22} {'before':>15} {'after':>15} {'change':>9}")
    for label, key, spread_key in ROWS:
        b, a = before.get(key), after.get(key)
        if b is None or a is None:
            continue
        lo = _cell(b, before.get(spread_key) if spread_key else None)
        hi = _cell(a, after.get(spread_key) if spread_key else None)
        if key == "first_request_ms":
            lo += "*" if before.get("first_request_compiled") else ""
            hi += "*" if after.get("first_request_compiled") else ""
        print(f"{label:22} {lo:>15} {hi:>15} {_change(b, a):>9}")

    # In the table, because the number that gets misread is the one pasted into
    # a PR without the surrounding output.
    if before.get("first_request_compiled") or after.get("first_request_compiled"):
        print(
            "\n* compiled from a cold Inductor cache: a compile time, not a "
            "regression.\n  Editing the graph invalidates the cache, so this lands "
            "on the changed side.\n  Re-run it in a fresh process."
        )

    print()
    ok = True
    for rec, name in ((before, "before"), (after, "after")):
        if rec.get("output_deterministic") is False:
            print(f"! {name} was not deterministic across repeats: {rec['output_sha']}")
            ok = False
    if before["output_sha"] == after["output_sha"]:
        print(f"output: identical ({before['output_sha']})")
    else:
        print(f"output: CHANGED {before['output_sha']} -> {after['output_sha']}")
        if before.get("output_text") and after.get("output_text"):
            print("  " + _first_difference(before["output_text"], after["output_text"]))
        print("  Explain this before claiming a speedup.")
        ok = False
    if before["tokens_out"] != after["tokens_out"]:
        print(
            f"! token counts differ ({before['tokens_out']} vs "
            f"{after['tokens_out']}); e2e is not comparable"
        )

    print(f"\nspeedup: {before['tpot_ms'] / after['tpot_ms']:.2f}x on TPOT")
    return 0 if ok else 1


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
    r.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 is greedy, which makes the output hash a correctness check. "
        "Must match `walnut profile --temperature` for the trace to describe "
        "the same workload.",
    )
    r.add_argument("--seed", type=int, default=None, help="seed for sampled runs")
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
