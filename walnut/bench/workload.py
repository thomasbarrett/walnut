"""What gets sent, and when: prompts, the arrival process, and the SLOs.

Kept apart from the transports that drive them: these decide whether a number
is honest, and they are testable without a GPU or a server.
"""

from __future__ import annotations

import random
from typing import Any

from walnut.bench.errors import BenchError
from walnut.bench.metrics import SLO_METRICS

PROMPT = "Explain how a transformer works."


class WorkloadError(BenchError):
    """A workload that cannot be built, or SLOs that cannot be parsed."""


def arrival_delays(
    count: int, rate: float, burstiness: float, rng: random.Random
) -> list[float]:
    """Gaps between consecutive submissions, in seconds.

    A gamma process with shape ``burstiness`` and mean ``1/rate``; at 1.0 that
    is Poisson. Below 1.0 arrivals clump, above 1.0 they even out.

    Firing everything at once measures a saturated engine and nothing else.
    Real traffic arrives and queues, and the queueing is most of the tail.
    """
    if rate == float("inf"):
        return [0.0] * count
    theta = 1.0 / (rate * burstiness)
    return [rng.gammavariate(burstiness, theta) for _ in range(count)]


def random_prompts(
    tokenizer: Any, count: int, input_len: int, range_ratio: float, rng: random.Random
) -> list[str]:
    """Synthetic prompts of roughly ``input_len`` tokens each.

    Random ids decoded back to text. The round trip is approximate, so records
    report the lengths actually measured rather than the ones asked for.
    ``range_ratio`` spreads lengths over ``[(1-r)*len, (1+r)*len]``; a run where
    every prompt is the same length hides that a mixed batch pads to its
    longest member.
    """
    vocab = tokenizer.vocab_size
    special = set(tokenizer.all_special_ids or ())
    lo = max(1, int(input_len * (1 - range_ratio)))
    hi = max(lo, int(input_len * (1 + range_ratio)))
    prompts = []
    for _ in range(count):
        length = rng.randint(lo, hi)
        ids = []
        while len(ids) < length:
            candidate = rng.randrange(vocab)
            if candidate not in special:
                ids.append(candidate)
        prompts.append(tokenizer.decode(ids))
    return prompts


def build_workload(
    dataset: str,
    num_prompts: int,
    prompt: str,
    input_len: int,
    range_ratio: float,
    tokenizer_id: str | None,
    rng: random.Random,
) -> list[str]:
    """The prompts a run will send, and nothing about how they are paced."""
    if dataset == "fixed":
        return [prompt] * num_prompts
    if tokenizer_id is None:
        raise WorkloadError("--dataset random needs --tokenizer or --model")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    return random_prompts(tokenizer, num_prompts, input_len, range_ratio, rng)


def goodput_config(pairs: list[str] | None) -> dict[str, float]:
    """Parse ``KEY:MILLISECONDS`` pairs into seconds. Repeated or
    comma-separated, both read the same."""
    config: dict[str, float] = {}
    for value in pairs or ():
        for pair in value.split(","):
            if not pair.strip():
                continue
            key, _, limit = pair.partition(":")
            key = key.strip()
            if key not in SLO_METRICS or not limit.strip():
                raise WorkloadError(
                    f"--goodput takes KEY:MILLISECONDS with KEY in "
                    f"{', '.join(SLO_METRICS)}; got {pair!r}"
                )
            config[key] = float(limit) / 1e3
    return config
