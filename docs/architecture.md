# Architecture

walnut is split so that the HTTP layer never depends on model internals. The
seam between them is the [`Engine`](reference.md#walnut.engine.Engine)
interface.

```
walnut chat  ──HTTP──▶  server (FastAPI)  ──▶  Engine  ──▶  model (PyTorch)
   client.py             server.py              engine.py
```

## The layers

- **`walnut/cli.py`** — the `walnut` command (`serve`, `chat`), built on Typer.
- **`walnut/server.py`** — the OpenAI-compatible FastAPI app. Translates HTTP
  requests into `Engine` calls and formats responses (including SSE streaming).
- **`walnut/engine.py`** — the engine interface plus `Message`,
  `GenerationConfig`, and the `EchoEngine` placeholder.
- **`walnut/client.py`** — a small OpenAI-compatible client used by `walnut chat`.

## The engine seam

`serve` loads a model through
[`load_model`](reference.md#walnut.engine.load_model), which returns an
`Engine`. Today it returns `EchoEngine`, which echoes the last user turn so the
server and client can be exercised without a real model. Because the server
depends only on the interface, a real engine plugs in without touching the HTTP
layer:

1. Implement an `Engine` subclass — load weights in `__init__` (or a
   classmethod), set `model_id`, and produce tokens in `generate`.
2. Optionally override `stream` for token-by-token streaming; the default yields
   the whole completion in one chunk.
3. Return it from `load_model` in place of `EchoEngine`.
