"""The continuous-batching loop, driven by a model stand-in.

The stand-in generates ``token + 1`` per step out of a real `KVCache`, so a
sequence's output is a function of its own prompt and nothing else: a row that
read another row's slot, or its own at the wrong position, shows up as a wrong
token rather than as a wrong number.
"""

import threading

import pytest
import torch

from walnut.graph import buckets
from walnut.layers.attention import Attention
from walnut.sampler import Sampler, SamplingParams
from walnut.scheduler import Request, RequestError, Scheduler

VOCAB = 64


class _StepModel:
    """Emits ``(last token + 1) % VOCAB``, through a real KV cache."""

    def __init__(self) -> None:
        self.attn = Attention(num_heads=1, num_kv_heads=1, head_dim=8)
        self.sampler = Sampler()
        self.eos_token_id = None
        self.steps = 0
        self.fail = False
        self.crash = False

    def make_cache(self, max_batch_size, max_seq_len):
        return [self.attn.make_cache(max_batch_size, max_seq_len, torch.float32, None)]

    def __call__(self, input_ids, positions, cache):
        if self.crash:
            raise BaseException("the loop died")  # noqa: TRY002 - escapes `_step`
        if self.fail:
            raise RuntimeError("the step broke")
        self.steps += 1
        batch, seq = input_ids.shape
        # Write the token into the cache and read it back out of the slot the
        # positions name, so a mis-slotted write becomes a wrong token.
        value = input_ids.float()[..., None, None].expand(batch, seq, 1, 8)
        _, values = cache[0].update(positions, torch.zeros_like(value), value.clone())
        if positions.ndim == 1:
            last = values[:, positions[-1], 0, 0]
        else:
            rows = torch.arange(batch)
            last = values[rows, positions[:, 0], 0, 0]
        logits = torch.zeros(batch, seq, VOCAB)
        logits[:, -1, :] = torch.nn.functional.one_hot(
            (last.long() + 1) % VOCAB, VOCAB
        ).float()
        return logits


def _scheduler(max_batch_size=4, max_seq_len=64):
    return Scheduler(
        _StepModel(),
        torch.device("cpu"),
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        cuda_graph=False,
        compile=False,
    )


def _request(start, tokens=4):
    return Request(
        prompt=torch.tensor([[start]]),
        params=SamplingParams(max_new_tokens=tokens, temperature=0.0),
        stop_ids=frozenset(),
    )


def _expected(start, tokens=4):
    return [(start + i + 1) % VOCAB for i in range(tokens)]


def _drain(scheduler, requests):
    """Run every request at once and return each one's tokens."""
    out: dict[int, list[int]] = {}

    def drive(index, request):
        scheduler.submit(request)
        out[index] = list(request.stream())

    threads = [
        threading.Thread(target=drive, args=(i, r)) for i, r in enumerate(requests)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [out[i] for i in range(len(requests))]


def test_a_lone_request_generates_its_own_sequence():
    scheduler = _scheduler()
    try:
        assert _drain(scheduler, [_request(5)]) == [_expected(5)]
    finally:
        scheduler.close()


def test_batched_requests_do_not_read_each_others_slots():
    """The bug this guards: a row indexing the pool by anything but its own
    slot, which a uniform batch would hide."""
    scheduler = _scheduler()
    starts = [3, 17, 41, 58]
    try:
        got = _drain(scheduler, [_request(s) for s in starts])
    finally:
        scheduler.close()
    assert got == [_expected(s) for s in starts]


def test_more_requests_than_slots_still_all_run():
    scheduler = _scheduler(max_batch_size=2)
    starts = [1, 2, 3, 4, 5, 6]
    try:
        got = _drain(scheduler, [_request(s) for s in starts])
    finally:
        scheduler.close()
    assert got == [_expected(s) for s in starts]


def test_a_finished_request_frees_its_slot():
    scheduler = _scheduler()
    try:
        _drain(scheduler, [_request(5), _request(9)])
        assert scheduler.running == {}
        assert scheduler.free == list(range(4))
    finally:
        scheduler.close()


def test_a_stop_token_ends_the_stream_after_yielding_it():
    scheduler = _scheduler()
    request = Request(
        prompt=torch.tensor([[5]]),
        params=SamplingParams(max_new_tokens=8, temperature=0.0),
        stop_ids=frozenset({8}),
    )
    try:
        scheduler.submit(request)
        assert list(request.stream()) == [6, 7, 8]
    finally:
        scheduler.close()


def test_abandoning_a_stream_frees_the_slot():
    """A disconnected client must stop occupying a slot, not decode to its
    token limit with nowhere to put the tokens."""
    scheduler = _scheduler()
    request = _request(5, tokens=32)
    try:
        scheduler.submit(request)
        stream = request.stream()
        next(stream)
        stream.close()
        scheduler.close()
        assert scheduler.running == {}
        assert scheduler.free == list(range(4))
    finally:
        scheduler.close()


def test_a_request_longer_than_the_pool_is_rejected():
    scheduler = _scheduler(max_seq_len=16)
    try:
        request = Request(
            prompt=torch.zeros(1, 12, dtype=torch.long),
            params=SamplingParams(max_new_tokens=8),
            stop_ids=frozenset(),
        )
        try:
            scheduler.submit(request)
        except RequestError as exc:
            assert "16-token context" in str(exc)
        else:
            raise AssertionError("an oversized request was accepted")
    finally:
        scheduler.close()


def test_a_failed_step_reaches_the_requests_it_broke():
    scheduler = _scheduler()
    request = _request(5, tokens=8)
    scheduler.submit(request)
    stream = request.stream()
    next(stream)
    scheduler.model.fail = True  # break the next step
    try:
        list(stream)
    except Exception as exc:
        assert not isinstance(exc, StopIteration)
    else:
        raise AssertionError("a broken step ended the stream as if it succeeded")
    finally:
        scheduler.close()


def test_buckets_are_powers_of_two_up_to_the_limit():
    assert buckets(8) == [1, 2, 4, 8]
    assert buckets(1) == [1]
    assert buckets(6) == [1, 2, 4, 6]  # the limit itself, not the next power
    assert buckets(1024, smallest=256) == [256, 512, 1024]


def test_a_dead_loop_fails_its_requests_instead_of_hanging():
    """The bug this guards: the loop is the only thing that puts tokens on a
    request's queue, so a loop that dies without saying so leaves every caller
    blocked on a `get` that never returns — a server that stops answering
    rather than one that fails."""
    scheduler = _scheduler()
    request = _request(5, tokens=32)
    scheduler.submit(request)
    stream = request.stream()
    next(stream)
    # Not an Exception: `_step` handles those per-request. This kills the loop.
    scheduler.model.crash = True

    with pytest.raises(BaseException, match="the loop died"):
        list(stream)

    # And a request arriving after the loop is gone is refused, not queued.
    with pytest.raises(RequestError, match="stopped"):
        scheduler.submit(_request(9))


def test_a_seeded_request_gets_its_own_generator():
    """`sample_batch` keys a seeded row to its generator; without one built
    here the seed would be accepted and silently ignored."""
    scheduler = _scheduler()
    try:
        request = Request(
            prompt=torch.tensor([[5]]),
            params=SamplingParams(max_new_tokens=1, temperature=1.0, seed=7),
            stop_ids=frozenset(),
        )
        scheduler.submit(request)
        assert request.generator is not None
        list(request.stream())
    finally:
        scheduler.close()
