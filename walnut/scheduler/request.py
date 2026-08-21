"""What a caller asks for, and what the loop holds while it answers.

Three things that a request passes through, kept apart because they are known
at different moments. `SamplingParams` is settled before anything is submitted;
`Request` exists from submission until the last token leaves; `Sequence` exists
only between admission and retirement, because none of what it holds — a cache
row, a position — means anything until a request has been given one.

Nothing here runs a model, and nothing here decides what runs next. This is the
vocabulary both of those are stated in.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

#: Put on a request's queue to end its stream.
DONE = object()


class RequestError(RuntimeError):
    """A request that the scheduler could not run."""


@dataclass
class SamplingParams:
    """How a request wants its next token drawn, and when to stop.

    Plain data, deliberately: it is set by the caller, carried through the
    queue, and read on the sampling path, and it should be none of those
    layers' business to know how the others use it.
    """

    max_new_tokens: int = 20
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    stop_token_ids: tuple[int, ...] = field(default_factory=tuple)
    seed: int | None = None


@dataclass
class Request:
    """One sequence in flight, and the channel its tokens leave by."""

    prompt: torch.Tensor
    params: SamplingParams
    stop_ids: frozenset[int]
    generator: torch.Generator | None = None
    tokens: queue.SimpleQueue = field(default_factory=queue.SimpleQueue)
    cancelled: threading.Event = field(default_factory=threading.Event)

    def stream(self) -> Iterator[int]:
        """Yield this request's token ids until it finishes.

        Closing the iterator early cancels the request, so a disconnected
        client stops occupying a row at the next decode step.
        """
        try:
            while True:
                item = self.tokens.get()
                if item is DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self.cancelled.set()


@dataclass
class Sequence:
    """A request that holds a row: what the loop advances, one step at a time.

    Split from `Request` because none of it means anything until admission — a
    request still in the queue has no row to name and no position to be at.
    The loop deals in sequences; a caller only ever sees its request.
    """

    request: Request
    #: The cache row this sequence owns, until it retires.
    row: int
    #: The position its next token is written at.
    position: int = 0
    #: Prompt tokens prefilled so far. Equal to the prompt length the moment
    #: the sequence joins the batch.
    prefilled: int = 0
    produced: int = 0
