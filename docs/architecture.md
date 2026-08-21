# Architecture

walnut is split so that the HTTP layer never depends on model internals. The
seam between them is the [`Engine`](reference.md#walnut.engine.Engine)
interface.

```
walnut chat  ──HTTP──▶  server (FastAPI)  ──▶  Engine  ──▶  Scheduler  ──▶  model
   client.py             server.py            engine.py     scheduler/     (PyTorch)
                                                                │
                                                                ▼
                                                             runner/
```

## The layers

- **`walnut/cli.py`** — the `walnut` command (`serve`, `chat`), built on Typer.
- **`walnut/server.py`** — the OpenAI-compatible FastAPI app. Translates HTTP
  requests into `Engine` calls and formats responses (including SSE streaming).
- **`walnut/engine.py`** — the engine interface plus `Message`,
  `GenerationConfig`, `Completion`/`Usage`/`Stream`, and the default
  `TorchEngine`.
- **`walnut/scheduler/`** — the continuous batching loop: a pool of sequence
  rows, decoded as one batch, that requests join and leave as they arrive and
  finish. `request.py` holds the vocabulary a caller speaks in — a request,
  its sampling parameters, and the sequence it becomes once admitted.
- **`walnut/runner/`** — the device-facing half: CUDA graph capture for the
  decode step (`graphs.py`) and the token draw itself (`sampling.py`). The
  runner imports the scheduler's types; the scheduler imports nothing here.
- **`walnut/models/protocol.py`** — what the loop requires of a model: a
  forward pass, a cache it builds for itself, and a sampler.
- **`walnut/cache/`** — where decode state lives: what a layer remembers, which
  rows and pages a sequence holds, and how one forward pass addresses them.
- **`walnut/client.py`** — a small OpenAI-compatible client used by `walnut chat`.

## The engine seam

`serve` loads a model through
[`load_model`](reference.md#walnut.engine.load_model), which returns an
`Engine`. By default it returns a `TorchEngine`, which loads a Hugging Face
checkpoint and runs it in PyTorch. Because the server depends only on the
interface, an alternative engine plugs in without touching the HTTP layer:

1. Implement an `Engine` subclass — load weights in `__init__` (or a
   classmethod), set `model_id`, and produce a `Completion` (text plus its
   `Usage` token counts) from `complete`.
2. Optionally override `stream` for token-by-token streaming; it returns a
   `Stream`, which yields text deltas and carries the token counts behind them.
   The default yields the whole completion in one chunk.
3. Return it from `load_model` in place of `TorchEngine`.

A `Stream` counts tokens rather than deltas because the two do not always line
up: detokenization holds a piece back until it completes a character, so one
delta can carry two tokens. That is why the counts travel with the stream
instead of being left for the caller to infer.

## The batch

`TorchEngine` does not run requests itself. It hands each one to a
[`Scheduler`](reference.md#walnut.scheduler.Scheduler) and blocks on that
request's own token queue, so the HTTP layer keeps a thread per request while
the GPU sees a single batch.

The scheduler owns a pool holding two resources. A **row** is a sequence's place
in the batch — the recurrent state it advances and the block table it addresses
its cache through — and `max_batch_size` bounds those, because a decode step
runs them all. **Pages** are 256 tokens of key/value cache apiece, drawn from
one pool shared by every row; a request reserves as many as its prompt and
completion need and returns them when it retires. Only the second kind is sized
to demand, and only the second kind can be shared between two sequences whose
prompts agree.

One background thread runs the loop:

1. **Admit** at most one waiting request per iteration, if a row and enough
   pages are free; one that does not fit yet stays at the head of the queue.
   Its row is cleared and its prompt is prefilled alone, batch-1, against it.
   Prefills are not batched: a prompt already saturates the GPU, so grouping
   them would only pad to the longest one, and admitting a burst at once would
   stall every running stream at the same moment.
2. **Prefill** the admitted prompt `--prefill-chunk` tokens at a time, one
   chunk per iteration. Prefill runs alone, so a whole prompt in one pass is a
   gap in every stream already decoding — half a second at 16k tokens. The
   chunk bounds that gap without changing what the prompt costs.
3. **Decode** one step across every running sequence at once, padded up to the
   nearest captured batch size. The padding is what makes step 2 need care: a
   step runs whole buckets, so it steps the row being prefilled too, and the
   scheduler puts back the recurrent state that step advanced. The rows nobody
   holds are stepped as well, which is why the pool keeps one scratch page for
   their writes to land in.
4. **Retire** any sequence that hit a stop token, its token limit, or a client
   that went away, freeing its row and pages for the next admission.

Rows of that batch sit at different points in their own sequences, which is
what `torch.nn.attention.varlen_attn` is for, and what `walnut.layers.attention`
decodes through: it takes each row's context length as a tensor rather than as a
shape, so attention reads only as far as each sequence has actually got, and one
captured graph serves a row at any point in its sequence. The same call covers
prefill, where causal alignment makes a prompt's own mask. Paging is the other
half of the same argument: a row's keys are no longer a contiguous run, so the
kernel is handed the row's block table and follows it page by page.

Two things follow from batching that a single-request engine does not have.
Output is **batch-dependent**: the kernels are not batch-invariant, so a greedy
request can pick a different token when it runs alongside others than it would
alone — rows of the same batch agree with each other, but a batch of eight and a
batch of one may diverge on a near-tie. And a request's latency now depends on
its neighbours; see [Batch size](cli.md#batch-size) for what that trade costs.
