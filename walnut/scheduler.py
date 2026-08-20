"""Continuous batching: many sequences decoded as one batch.

Decode is memory-bound. A step at batch 1 reads every weight in the model to
produce a single token, so a second sequence riding along costs almost nothing
— the weights are already in flight. Batching is how a serving engine turns
that read into throughput, and `Scheduler` is what keeps the batch full.

The batch is continuous rather than static: a request joins at the next decode
step instead of waiting for the current group to finish, and a finished
sequence frees its slot the step it stops. `max_batch_size` is the number of
slots, which is what vLLM's ``--max-num-seqs`` and SGLang's
``--max-running-requests`` bound.

Prefill runs alone, one request at a time, into its own slot. Batching prefills
would mean padding to the longest prompt in the group and paying attention over
the padding; a prompt already saturates the GPU on its own, so there is nothing
to win there and a stall to lose. Decode is where the batch is.

It runs a chunk at a time, though, with a decode step between chunks
(``prefill_chunk``). Alone and unchunked, a prompt stalls every sequence
already decoding for as long as it takes — half a second at 16k tokens, which
each of those streams sees as a half-second gap between two of its tokens. The
chunk bounds that gap without changing what prefill costs in total.

The pool is preallocated: `max_batch_size` slots of `max_seq_len` tokens each,
allocated once at start. A request whose prompt and completion do not fit is
rejected rather than allowed to displace a running one.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from walnut.cache import CachePool
from walnut.graph import DecodeGraphs, buckets
from walnut.sampler import SamplingParams

logger = logging.getLogger("walnut.scheduler")

#: Put on a request's queue to end its stream.
_DONE = object()


class RequestError(RuntimeError):
    """A request that the scheduler could not run."""


@dataclass
class Request:
    """One sequence in flight, and the channel its tokens leave by."""

    prompt: torch.Tensor
    params: SamplingParams
    stop_ids: frozenset[int]
    generator: torch.Generator | None = None
    tokens: queue.SimpleQueue = field(default_factory=queue.SimpleQueue)
    cancelled: threading.Event = field(default_factory=threading.Event)

    def stream(self) -> Any:
        """Yield this request's token ids until it finishes.

        Closing the iterator early cancels the request, so a disconnected
        client stops occupying a slot at the next decode step.
        """
        try:
            while True:
                item = self.tokens.get()
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self.cancelled.set()


@dataclass
class Sequence:
    """A request that holds a slot: what the loop advances, one step at a time.

    Split from `Request` because none of it means anything until admission — a
    request still in the queue has no slot to name and no position to be at.
    The loop deals in sequences; a caller only ever sees its request.
    """

    request: Request
    #: The cache slot this sequence owns, until it retires.
    slot: int
    #: The position its next token is written at.
    position: int = 0
    #: Prompt tokens prefilled so far. Equal to the prompt length the moment
    #: the sequence joins the batch.
    prefilled: int = 0
    produced: int = 0


class Scheduler:
    """Runs a pool of sequence slots as one continuously batched decode loop.

    Owns a background thread; `submit` hands it a `Request` and returns, and
    the request's tokens arrive on its own queue as they are sampled.
    """

    def __init__(
        self,
        model: Any,
        device: torch.device,
        max_batch_size: int = 8,
        max_seq_len: int = 8192,
        cuda_graph: bool = True,
        compile: bool = True,
        autotune: bool = True,
        prefill_chunk: int = 2048,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if prefill_chunk < 1:
            raise ValueError("prefill_chunk must be at least 1")
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.prefill_chunk = prefill_chunk
        self.cuda_graph = cuda_graph and device.type == "cuda"
        self.compile = compile
        self.autotune = autotune

        self.pool: CachePool = model.make_cache(max_batch_size, max_seq_len)
        # The pool is decoded against from the first step: its slots hold
        # zeros, which is what "no context yet" means to a recurrent state.
        self.pool.prime()

        self.sizes = buckets(max_batch_size)
        self.views = {size: self.pool.view(0, size) for size in self.sizes}
        # Built once rather than per prefill: a view allocates its own index
        # tensors, and rebuilding one per request puts that on the time-to-
        # first-token path.
        self.slot_views = [self.pool.slot(slot) for slot in range(max_batch_size)]
        self.token = torch.zeros(max_batch_size, 1, dtype=torch.long, device=device)
        self.position = torch.zeros(max_batch_size, 1, dtype=torch.long)

        # The recurrent state a decode step would advance under a chunked
        # prefill, per slot, alongside one scratch copy to hold it in. Both are
        # built once: a prefill chunk is on the time-to-first-token path, and
        # only one prefill is ever in flight, so one copy serves whichever slot
        # holds it.
        self.carried = [self.pool.carried(slot) for slot in range(max_batch_size)]
        self.saved = [torch.empty_like(tensor) for tensor in self.carried[0]]

        self.incoming: queue.Queue[Request | None] = queue.Queue()
        self.running: dict[int, Sequence] = {}
        #: The sequence being prefilled, if any. At most one: its chunks run
        #: between decode steps, and a second would only lengthen both.
        self.prefilling: Sequence | None = None
        self.decode_forward: Any = model
        self.graphs: DecodeGraphs | None = None
        self._started = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        #: Set once the loop has stopped, for any reason. A scheduler that is
        #: not running cannot serve anything, so this turns a later `submit`
        #: into an error rather than a request that waits forever.
        self._stopped: BaseException | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Compile, capture, and bring the loop up. Idempotent.

        Doing this before the first request rather than during it keeps the
        one-off cost — a compile per bucket, and a capture per bucket — out of
        a request's latency.
        """
        with self._lock:
            if self._stopped is not None:
                raise RequestError("the scheduler has stopped") from self._stopped
            if self._thread is not None:
                return
            self._warmup()
            self._thread = threading.Thread(
                target=self._run, name="walnut-scheduler", daemon=True
            )
            self._thread.start()
        self._started.wait()

    def close(self) -> None:
        """Stop the loop once the running batch drains. Idempotent, and final:
        a scheduler does not restart, because starting one compiles and
        captures a graph per bucket."""
        if self._thread is None:
            return
        self.incoming.put(None)
        self._thread.join()
        self._thread = None

    def _warmup(self) -> None:
        """Compile the decode step and capture a graph per bucket.

        Compilation happens against the pool, which already holds decode-shaped
        state, so dynamo records the decode branch without a prefill having run
        (see `iter_generate`, which has to prefill first for the same reason).
        """
        if self.compile:
            mode = "max-autotune-no-cudagraphs" if self.autotune else None
            self.decode_forward = torch.compile(self.model, mode=mode)
        if self.cuda_graph:
            # No snapshot: the pool is empty at this point and every slot is
            # reset before a sequence takes it, so there is nothing capture
            # could disturb — and a snapshot would be a second whole pool.
            self.graphs = DecodeGraphs(
                self.decode_forward,
                self.pool,
                self.device,
                self.max_batch_size,
                restore=False,
            )

    # -- submission --------------------------------------------------------

    def submit(self, request: Request) -> Request:
        """Queue ``request`` and return it; its tokens arrive on its queue."""
        self.start()
        if request.generator is None and request.params.seed is not None:
            # A seeded request draws from its own generator; without one it
            # would silently share the batch's draw and stop being reproducible.
            request.generator = torch.Generator(device=self.device).manual_seed(
                request.params.seed
            )
        wanted = request.prompt.shape[1] + request.params.max_new_tokens
        if wanted > self.max_seq_len:
            raise RequestError(
                f"{request.prompt.shape[1]} prompt tokens plus "
                f"{request.params.max_new_tokens} generated exceeds the "
                f"{self.max_seq_len}-token context; raise --max-seq-len"
            )
        self.incoming.put(request)
        return request

    # -- the loop ----------------------------------------------------------

    @property
    def _idle(self) -> bool:
        """Nothing in flight: no sequence decoding, no prompt part-prefilled."""
        return not self.running and self.prefilling is None

    @torch.no_grad()
    def _run(self) -> None:
        self._started.set()
        error: BaseException | None = None
        stopping = False
        try:
            while not (stopping and self._idle):
                if not self._admit():
                    stopping = True
                if self.prefilling is not None:
                    self._prefill_chunk()
                if self.running:
                    self._step()
        except BaseException as exc:  # noqa: BLE001 — re-raised by `_drain`
            error = exc
            logger.exception("scheduler_loop_failed")
        finally:
            self._drain(error)

    def _drain(self, error: BaseException | None) -> None:
        """Close every stream the loop will not be serving.

        This loop is the only thing that ever puts a token on a request's
        queue, so a loop that leaves — cleanly or otherwise — without saying so
        leaves every caller blocked on a `get` that will not return: a server
        that stops answering rather than one that fails. Whatever is running or
        waiting gets the reason instead.
        """
        reason = error or RequestError("the scheduler has stopped")
        self._stopped = reason
        if self.prefilling is not None:
            self._abandon(self.prefilling, reason)
        for sequence in list(self.running.values()):
            self._retire(sequence, error)
        while True:
            try:
                waiting = self.incoming.get_nowait()
            except queue.Empty:
                break
            if waiting is not None:
                self._close(waiting, reason)

    def _admit(self) -> bool:
        """Take at most one waiting request into the batch; False to stop.

        One at a time on purpose, and none at all while another is still
        prefilling: chunks and decode steps share the loop, so a second prompt
        would only interleave with the first and make both slower to answer.
        """
        if self.prefilling is not None or not self.pool.free:
            return True
        blocking = not self.running
        try:
            request = self.incoming.get(block=blocking)
        except queue.Empty:
            return True
        if request is None:
            return False
        if not request.cancelled.is_set():
            self._begin_prefill(request)
        return True

    def _begin_prefill(self, request: Request) -> None:
        """Give ``request`` a fresh slot; its prompt runs from the next chunk."""
        slot = self.pool.reserve()
        assert slot is not None  # `_admit` checked it had one
        self.pool.reset(slot)
        self.prefilling = Sequence(request, slot)

    def _prefill_chunk(self) -> None:
        """Run the next ``prefill_chunk`` of the admitted prompt.

        Returns to the loop between chunks, so every sequence already decoding
        gets a token in the gap. The prompt's own cost does not change: the
        chunk carries the recurrent state forward and attends over everything
        written before it, which is the same arithmetic the whole prompt in one
        pass does, and at 2048 it measures the same too.
        """
        sequence = self.prefilling
        assert sequence is not None
        request = sequence.request
        if request.cancelled.is_set():
            # A client that left mid-prefill frees its slot now rather than
            # after the remaining chunks it will not read.
            self._abandon(sequence, None)
            return
        slot = sequence.slot
        start = sequence.prefilled
        chunk = request.prompt[:, start : start + self.prefill_chunk]
        try:
            positions = torch.arange(start, start + chunk.shape[1], device=self.device)
            logits = self.model(chunk, positions=positions, cache=self.slot_views[slot])
            sequence.prefilled = start + chunk.shape[1]
            if sequence.prefilled < request.prompt.shape[1]:
                return
            token = self.model.sampler(logits[:, -1], request.params, request.generator)
            self.token[slot].copy_(token[0])
            token_id = int(token.item())
        except Exception as exc:  # deliver the failure to its own caller only
            self._abandon(sequence, exc)
            return

        self.prefilling = None
        sequence.position = sequence.prefilled
        self.running[slot] = sequence
        self._deliver(sequence, token_id)

    def _abandon(self, sequence: Sequence, error: BaseException | None) -> None:
        """Drop a part-prefilled sequence, freeing the slot it never filled."""
        self.prefilling = None
        self.pool.release(sequence.slot)
        self._close(sequence.request, error)

    def _bucket(self) -> int:
        """The smallest captured batch size covering every live slot.

        Rows above the live set run too: free for a slot nobody holds, and
        paid for by `_holding` for one part way through a prefill.
        """
        return min(size for size in self.sizes if size > max(self.running))

    @contextmanager
    def _holding(self, sequence: Sequence | None) -> Iterator[None]:
        """Keep a part-prefilled row's recurrent state across a decode step.

        A step runs whole buckets, so a row part way through a prefill is
        stepped along with the rest. A positional write survives that — the
        row is aimed at the position its next chunk overwrites — but recurrent
        state has nowhere to be aimed: a step advances it wherever it is
        pointed, and the prompt's context is gone. So it is saved here and put
        back. Two `_foreach_copy_` calls, and only while a prefill is in
        flight.
        """
        # Nothing in flight, or nothing to hold: a cache that writes only by
        # position has no state a stray step can carry off.
        if sequence is None or not self.saved:
            yield
            return
        torch._foreach_copy_(self.saved, self.carried[sequence.slot])
        try:
            yield
        finally:
            torch._foreach_copy_(self.carried[sequence.slot], self.saved)

    def _forward(self, size: int) -> list[int]:
        """Decode ``size`` rows from `position` and `token`, one token each."""
        rows = self.position[:size]
        if self.graphs is not None:
            logits = self.graphs.replay(self.token[:size], rows)
        else:
            positions = rows.to(self.device)
            logits = self.decode_forward(
                self.token[:size], positions=positions, cache=self.views[size]
            )
        # Padding rows sample under a live row's settings, so a batch whose
        # requests agree stays a single sampling call. They take no generator:
        # a seeded draw belongs to the one request that asked for it.
        spare = next(iter(self.running.values())).request.params
        params = []
        gens = []
        for slot in range(size):
            sequence = self.running.get(slot)
            params.append(sequence.request.params if sequence else spare)
            gens.append(sequence.request.generator if sequence else None)
        sampled = self.model.sampler.sample_batch(logits[:, -1], params, gens)
        self.token[:size].copy_(sampled)
        return sampled[:, 0].tolist()

    def _step(self) -> None:
        """One decode step over every running sequence."""
        size = self._bucket()
        self.position[:size].zero_()
        for slot, sequence in self.running.items():
            self.position[slot] = sequence.position

        held = self.prefilling
        held = held if held is not None and held.slot < size else None
        if held is not None:
            # The step writes a token into the prefilling row too. Aim it at
            # the position the next chunk overwrites, so the write is undone.
            self.position[held.slot] = held.prefilled

        try:
            with self._holding(held):
                ids = self._forward(size)
        except Exception as exc:
            for sequence in list(self.running.values()):
                self._retire(sequence, exc)
            return

        for slot, sequence in list(self.running.items()):
            sequence.position += 1
            self._deliver(sequence, int(ids[slot]))

    # -- delivery ----------------------------------------------------------

    def _deliver(self, sequence: Sequence, token_id: int) -> None:
        """Hand one token to its caller and retire the sequence if it is done."""
        request = sequence.request
        sequence.produced += 1
        request.tokens.put(token_id)
        done = (
            token_id in request.stop_ids
            or sequence.produced >= request.params.max_new_tokens
            or request.cancelled.is_set()
        )
        if done:
            self._retire(sequence)

    def _retire(self, sequence: Sequence, error: BaseException | None = None) -> None:
        """Free a sequence's slot and close its stream."""
        if self.running.pop(sequence.slot, None) is None:
            return
        self.pool.release(sequence.slot)
        self._close(sequence.request, error)

    @staticmethod
    def _close(request: Request, error: BaseException | None) -> None:
        """End a request's stream, with the reason if there was one."""
        if error is not None:
            request.tokens.put(error)
        request.tokens.put(_DONE)
