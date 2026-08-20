"""What gets sent, and when: the workload shapes, the arrival process, the SLOs.

Kept apart from the transports that drive them: these decide whether a number
is honest, and they are testable without a GPU or a server.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from walnut.bench.errors import BenchError
from walnut.bench.metrics import SLO_METRICS


class WorkloadError(BenchError):
    """A workload that cannot be built, or SLOs that cannot be parsed."""


@dataclass(frozen=True)
class Shape:
    """How much prompt, and how much generation.

    The primary axis of any serving measurement, and the one walnut is most
    sensitive to: `Scheduler` prefills one request at a time and alone, so the
    cost of a prompt is paid by every stream already running — in chunks of
    ``prefill_chunk`` rather than all at once, which bounds the gap without
    moving the cost. Between `reasoning` and `agentic` the prefill:decode work
    ratio moves by two orders of magnitude, and no conclusion drawn at one end
    transfers to the other.

    Named rather than assembled from flags so the record says which workload it
    measured, and two runs can be checked against each other instead of trusted.
    """

    name: str
    #: The longest prompt the shape sends, not its average — see ``jitter``.
    input_len: int
    output_len: int
    #: How far below ``input_len`` prompts may fall, as a fraction: 0.2 spreads
    #: them over ``[0.8 * input_len, input_len]``. One-sided, following
    #: InferenceMAX, so the shape's name states a ceiling rather than a mean —
    #: "at most 1024 in" is a claim a reader can check against a record.
    #: Nonzero because a run where every prompt is the same length never shows
    #: that a mixed batch pads to its longest member.
    jitter: float
    what: str


#: 20%, so prompts span 80-100% of the named length. InferenceMAX's figure.
JITTER = 0.2

#: Every size here is lifted from a published benchmark rather than invented:
#: `chat` and `reasoning` from SemiAnalysis's InferenceMAX, `rag` and `agentic`
#: from Luminal's reports. No single publisher uses this exact set — there is
#: no industry standard to follow — but each row can be read beside the source
#: it came from. Output length is pinned on the prefill-heavy shapes so that
#: generation cost cannot mask the prompt cost, which is the convention both
#: sources share.
SHAPES: dict[str, Shape] = {
    "chat": Shape("chat", 1024, 1024, JITTER, "a conversational turn"),
    "rag": Shape("rag", 4096, 256, JITTER, "retrieval: prefill-heavy, short answer"),
    "reasoning": Shape(
        "reasoning", 1024, 8192, JITTER, "a long scratchpad: decode-bound"
    ),
    "agentic": Shape(
        "agentic",
        16384,
        256,
        JITTER,
        "full history and tool schemas: prefill is the cost",
    ),
}

#: The cheapest shape anyone actually runs. There is no smaller default worth
#: having: a short synthetic prompt is nearly all decode, and a regression gate
#: that only guards decode passes changes that ruin prefill.
DEFAULT_SHAPE = "chat"


def resolve_shape(name: str) -> Shape:
    if name not in SHAPES:
        raise WorkloadError(f"--shape takes one of {', '.join(SHAPES)}; got {name!r}")
    return SHAPES[name]


def arrival_delays(count: int, rate: float, rng: random.Random) -> list[float]:
    """Gaps between consecutive submissions, in seconds.

    A Poisson process: exponential gaps with mean ``1/rate``. Fixed, not a
    knob. The general form is a gamma process whose shape parameter tunes how
    much arrivals clump, and there is no trace here to calibrate that shape
    against — so every value but Poisson would be a number chosen to produce a
    result rather than to describe traffic.

    Firing everything at once measures a saturated engine and nothing else.
    Real traffic arrives and queues, and the queueing is most of the tail.
    """
    if rate == float("inf"):
        return [0.0] * count
    return [rng.expovariate(rate) for _ in range(count)]


def load_tokenizer(tokenizer_id: str | None) -> Any:
    if tokenizer_id is None:
        raise WorkloadError(
            "prompts are generated and need a tokenizer; pass --tokenizer or --model"
        )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_id)


def build_workload(
    shape: Shape, count: int, tokenizer: Any, rng: random.Random
) -> list[str]:
    """The prompts a run will send, and nothing about how they are paced.

    Random ids decoded back to text. The round trip is approximate, so records
    report the prompt lengths actually measured rather than the ones asked for.
    """
    vocab = tokenizer.vocab_size
    special = set(tokenizer.all_special_ids or ())
    lo = max(1, int(shape.input_len * (1 - shape.jitter)))
    hi = max(lo, shape.input_len)
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
