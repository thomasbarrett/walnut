# Getting started

## Requirements

- [uv](https://docs.astral.sh/uv/) for package and environment management
- Python 3.12+ (installed automatically by uv via `.python-version`)

## Install

```bash
uv sync --extra cpu --dev
```

## Serve a model

Serve a model (a Hugging Face id or a local path) behind an OpenAI-compatible
API:

```bash
uv run walnut serve Qwen/Qwen3.5-0.8B --host 0.0.0.0 --port 8000
```

The server exposes `/v1` — see the [HTTP API](http-api.md) reference.

## Chat with it

In another terminal:

```bash
# One-shot
uv run walnut chat --quick "Hello!"

# Interactive REPL
uv run walnut chat
```

`chat` defaults to the first model the backend reports and talks to
`http://127.0.0.1:8000/v1` unless you pass `--url`. Any OpenAI client works too —
point the `openai` SDK at `http://127.0.0.1:8000/v1`.
