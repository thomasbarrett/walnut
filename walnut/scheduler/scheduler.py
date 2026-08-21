"""The loop itself: admit, prefill a chunk, decode a step, deliver, retire."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch

from walnut.cache import CachePool, pages_for
from walnut.runner.graphs import DecodeGraphs, buckets
from walnut.scheduler.request import DONE, Request, RequestError, Sequence

if TYPE_CHECKING:
    from walnut.models.protocol import CausalLM

logger = logging.getLogger("walnut.scheduler")


class Scheduler:
    """Runs a pool of sequence rows as one continuously batched decode loop.

    Owns a background thread; `submit` hands it a `Request` and returns, and
    the request's tokens arrive on its own queue as they are sampled.
    """

    def __init__(
        self,
        model: CausalLM,
        device: torch.device,
        max_batch_size: int = 8,
        max_seq_len: int = 8192,
        cuda_graph: bool = True,
        compile: bool = True,
        autotune: bool = True,
        prefill_chunk: int = 2048,
        kv_tokens: int | None = None,
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

        pages = None if kv_tokens is None else max(1, pages_for(kv_tokens))
        self.pool: CachePool = model.make_cache(max_batch_size, max_seq_len, pages)
        # The pool is decoded against from the first step: its rows hold
        # zeros, which is what "no context yet" means to a recurrent state.
        self.pool.prime()

        self.sizes = buckets(max_batch_size)
        self.views = {size: self.pool.view(0, size) for size in self.sizes}
        # Built once rather than per prefill: a view allocates its own index
        # tensors, and rebuilding one per request puts that on the time-to-
        # first-token path. A row's view outlives the sequences that hold it —
        # it names the row, not the pages, which change under it as each new
        # sequence reserves its own.
        self.row_views = [self.pool.row(row) for row in range(max_batch_size)]
        self.token = torch.zeros(max_batch_size, 1, dtype=torch.long, device=device)
        self.position = torch.zeros(max_batch_size, 1, dtype=torch.long)

        # The recurrent state a decode step would advance under a chunked
        # prefill, per row, alongside one scratch copy to hold it in. Both are
        # built once: a prefill chunk is on the time-to-first-token path, and
        # only one prefill is ever in flight, so one copy serves whichever row
        # holds it.
        self.carried = [self.pool.carried(row) for row in range(max_batch_size)]
        self.saved = [torch.empty_like(tensor) for tensor in self.carried[0]]

        self.incoming: queue.Queue[Request | None] = queue.Queue()
        self.running: dict[int, Sequence] = {}
        #: The sequence being prefilled, if any. At most one: its chunks run
        #: between decode steps, and a second would only lengthen both.
        self.prefilling: Sequence | None = None
        #: A request taken off the queue that the pool could not fit yet. It
        #: stays here rather than going back on the queue, so admission is
        #: strictly first-come: a short request behind a long one waits for it
        #: rather than overtaking it into the pages it was about to claim.
        self.waiting: Request | None = None
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
            # No snapshot: the pool is empty at this point and every row is
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
        if not self.pool.fits(wanted):
            # Refused now rather than queued: the pool empties as sequences
            # retire, and this one would still not fit an empty pool.
            raise RequestError(
                f"{wanted} tokens needs {pages_for(wanted)} pages of key/value "
                f"cache and the pool holds {self.pool.pages}; raise --kv-tokens"
            )
        self.incoming.put(request)
        return request

    # -- the loop ----------------------------------------------------------

    @property
    def _idle(self) -> bool:
        """Nothing in flight: nothing decoding, prefilling, or waiting for room."""
        return not self.running and self.prefilling is None and self.waiting is None

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
        if self.waiting is not None:
            self._close(self.waiting, reason)
            self.waiting = None
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

        A request the pool cannot fit stays in `waiting` and is retried each
        time round the loop, which is where it ends up once a running sequence
        retires and returns its pages. That cannot deadlock: a request only
        reaches here having passed `CachePool.fits`, so an empty pool always
        has room for it, and the loop only blocks on the queue while nothing
        is running — which is exactly when the pool is empty.
        """
        if self.prefilling is not None:
            return True
        if self.waiting is None:
            blocking = not self.running
            try:
                request = self.incoming.get(block=blocking)
            except queue.Empty:
                return True
            if request is None:
                return False
            self.waiting = request
        if self.waiting.cancelled.is_set() or self._begin_prefill(self.waiting):
            self.waiting = None
        return True

    def _begin_prefill(self, request: Request) -> bool:
        """Give ``request`` a row and its pages; False if the pool has neither.

        The reservation covers the prompt *and* the completion the request
        asked for, so a sequence never runs out of pages part way through
        decoding. That is what makes preemption unnecessary here, and it is
        also what a slot pool was doing implicitly — except that it reserved
        `max_seq_len` for every request rather than what the request asked for.
        """
        wanted = request.prompt.shape[1] + request.params.max_new_tokens
        row = self.pool.reserve(wanted)
        if row is None:
            return False
        self.pool.reset(row)
        self.prefilling = Sequence(request, row)
        return True

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
            # A client that left mid-prefill frees its row now rather than
            # after the remaining chunks it will not read.
            self._abandon(sequence, None)
            return
        row = sequence.row
        start = sequence.prefilled
        chunk = request.prompt[:, start : start + self.prefill_chunk]
        try:
            positions = torch.arange(start, start + chunk.shape[1], device=self.device)
            logits = self.model(chunk, positions=positions, cache=self.row_views[row])
            sequence.prefilled = start + chunk.shape[1]
            if sequence.prefilled < request.prompt.shape[1]:
                return
            token = self.model.sampler(logits[:, -1], request.params, request.generator)
            self.token[row].copy_(token[0])
            token_id = int(token.item())
        except Exception as exc:  # deliver the failure to its own caller only
            self._abandon(sequence, exc)
            return

        self.prefilling = None
        sequence.position = sequence.prefilled
        self.running[row] = sequence
        self._deliver(sequence, token_id)

    def _abandon(self, sequence: Sequence, error: BaseException | None) -> None:
        """Drop a part-prefilled sequence, freeing the row it never filled."""
        self.prefilling = None
        self.pool.release(sequence.row)
        self._close(sequence.request, error)

    def _bucket(self) -> int:
        """The smallest captured batch size covering every live row.

        Rows above the live set run too: free for a row nobody holds, and
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
        torch._foreach_copy_(self.saved, self.carried[sequence.row])
        try:
            yield
        finally:
            torch._foreach_copy_(self.carried[sequence.row], self.saved)

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
        for row in range(size):
            sequence = self.running.get(row)
            params.append(sequence.request.params if sequence else spare)
            gens.append(sequence.request.generator if sequence else None)
        sampled = self.model.sampler.sample_batch(logits[:, -1], params, gens)
        self.token[:size].copy_(sampled)
        return sampled[:, 0].tolist()

    def _step(self) -> None:
        """One decode step over every running sequence."""
        size = self._bucket()
        self.position[:size].zero_()
        for row, sequence in self.running.items():
            self.position[row] = sequence.position

        held = self.prefilling
        held = held if held is not None and held.row < size else None
        if held is not None:
            # The step writes a token into the prefilling row too. Aim it at
            # the position the next chunk overwrites, so the write is undone.
            self.position[held.row] = held.prefilled

        try:
            with self._holding(held):
                ids = self._forward(size)
        except Exception as exc:
            for sequence in list(self.running.values()):
                self._retire(sequence, exc)
            return

        for row, sequence in list(self.running.items()):
            sequence.position += 1
            self._deliver(sequence, int(ids[row]))

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
        """Free a sequence's row and pages, and close its stream."""
        if self.running.pop(sequence.row, None) is None:
            return
        self.pool.release(sequence.row)
        self._close(sequence.request, error)

    @staticmethod
    def _close(request: Request, error: BaseException | None) -> None:
        """End a request's stream, with the reason if there was one."""
        if error is not None:
            request.tokens.put(error)
        request.tokens.put(DONE)
