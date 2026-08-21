"""Continuous batching: many sequences decoded as one batch.

Decode is memory-bound. A step at batch 1 reads every weight in the model to
produce a single token, so a second sequence riding along costs almost nothing
— the weights are already in flight. Batching is how a serving engine turns
that read into throughput, and `Scheduler` is what keeps the batch full.

The batch is continuous rather than static: a request joins at the next decode
step instead of waiting for the current group to finish, and a finished
sequence frees its row the step it stops. `max_batch_size` is the number of
rows, which is what vLLM's ``--max-num-seqs`` and SGLang's
``--max-running-requests`` bound.

Prefill runs alone, one request at a time, into its own row. Batching prefills
would mean padding to the longest prompt in the group and paying attention over
the padding; a prompt already saturates the GPU on its own, so there is nothing
to win there and a stall to lose. Decode is where the batch is.

It runs a chunk at a time, though, with a decode step between chunks
(``prefill_chunk``). Alone and unchunked, a prompt stalls every sequence
already decoding for as long as it takes — half a second at 16k tokens, which
each of those streams sees as a half-second gap between two of its tokens. The
chunk bounds that gap without changing what prefill costs in total.

The pool is preallocated and holds two resources. A *row* is a sequence's
place in the batch — the recurrent state it advances, and the block table it
addresses its pages through — and `max_batch_size` bounds those because a
decode step runs them all. *Pages* are 256 tokens of key/value storage apiece,
drawn from one pool shared by every row, and a request takes only as many as
its prompt and completion need. A request that does not fit *yet* waits at the
head of the queue until a running one retires; one that could never fit is
rejected at `Scheduler.submit`, rather than waiting for room that will not come.
"""

from walnut.scheduler.request import (
    Request,
    RequestError,
    SamplingParams,
    Sequence,
)
from walnut.scheduler.scheduler import Scheduler

__all__ = [
    "Request",
    "RequestError",
    "SamplingParams",
    "Scheduler",
    "Sequence",
]
