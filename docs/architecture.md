# Architecture

walnut is split so that the HTTP layer never depends on model internals. The
seam between them is the [`Engine`](reference.md#walnut.engine.Engine)
interface.

```
walnut chat  ──HTTP──▶  server (FastAPI)  ──▶  Engine  ──▶  Scheduler  ──▶  model
   client.py             server.py            engine.py     scheduler.py   (PyTorch)
```

## The layers

- **`walnut/cli.py`** — the `walnut` command (`serve`, `chat`), built on Typer.
- **`walnut/server.py`** — the OpenAI-compatible FastAPI app. Translates HTTP
  requests into `Engine` calls and formats responses (including SSE streaming).
- **`walnut/engine.py`** — the engine interface plus `Message`,
  `GenerationConfig`, and the default `TorchEngine`.
- **`walnut/scheduler.py`** — the continuous batching loop: a pool of sequence
  slots, decoded as one batch, that requests join and leave as they arrive and
  finish.
- **`walnut/client.py`** — a small OpenAI-compatible client used by `walnut chat`.

## The engine seam

`serve` loads a model through
[`load_model`](reference.md#walnut.engine.load_model), which returns an
`Engine`. By default it returns a `TorchEngine`, which loads a Hugging Face
checkpoint and runs it in PyTorch. Because the server depends only on the
interface, an alternative engine plugs in without touching the HTTP layer:

1. Implement an `Engine` subclass — load weights in `__init__` (or a
   classmethod), set `model_id`, and produce tokens in `generate`.
2. Optionally override `stream` for token-by-token streaming; the default yields
   the whole completion in one chunk.
3. Return it from `load_model` in place of `TorchEngine`.

## The batch

`TorchEngine` does not run requests itself. It hands each one to a
[`Scheduler`](reference.md#walnut.scheduler.Scheduler) and blocks on that
request's own token queue, so the HTTP layer keeps a thread per request while
the GPU sees a single batch.

The scheduler owns a pool of `max_batch_size` sequence slots, preallocated for
`max_seq_len` tokens each. One background thread runs the loop:

1. **Admit** at most one waiting request per iteration. Its slot is cleared and
   its prompt is prefilled alone, batch-1, into that slot. Prefills are not
   batched: a prompt already saturates the GPU, so grouping them would only pad
   to the longest one, and admitting a burst at once would stall every running
   stream at the same moment.
2. **Decode** one step across every running sequence at once, padded up to the
   nearest captured batch size.
3. **Retire** any sequence that hit a stop token, its token limit, or a client
   that went away, freeing its slot for the next admission.

Rows of that batch sit at different points in their own sequences, which is
what `torch.nn.attention.varlen_attn` is for, and what `walnut.layers.attention`
decodes through: it takes each row's context length as a tensor rather than as a
shape, so attention reads only as far as each sequence has actually got, and one
captured graph serves a slot at any point in its sequence. The same call covers
prefill, where causal alignment makes a prompt's own mask.

Two things follow from batching that a single-request engine does not have.
Output is **batch-dependent**: the kernels are not batch-invariant, so a greedy
request can pick a different token when it runs alongside others than it would
alone — rows of the same batch agree with each other, but a batch of eight and a
batch of one may diverge on a near-tie. And a request's latency now depends on
its neighbours; see [Batch size](cli.md#batch-size) for what that trade costs.
