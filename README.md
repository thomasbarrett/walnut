# walnut

[![CI](https://github.com/thomasbarrett/walnut/actions/workflows/ci.yml/badge.svg)](https://github.com/thomasbarrett/walnut/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An inference engine, built on PyTorch, with an OpenAI-compatible API.

**[Documentation](https://thomasbarrett.github.io/walnut/)**

## Requirements

- [uv](https://docs.astral.sh/uv/) for package and environment management
- Python 3.12+ (installed automatically by uv via `.python-version`)

Optional developer tooling (Helm, hadolint, prek, gh) is captured in the
`Brewfile`:

```bash
brew bundle          # installs uv, helm, hadolint, prek, gh
```

## Getting started

```bash
# Install dependencies (including dev tools)
uv sync --extra cpu --dev

# Serve a model behind an OpenAI-compatible API (Hugging Face id or local path)
uv run walnut serve Qwen/Qwen3.5-0.8B --host 0.0.0.0 --port 8000

# Chat with it (defaults to the first model the backend reports)
uv run walnut chat --quick 'Hello!'
```

> Inference runs through `TorchEngine` (`walnut/engine.py:load_model`), which
> loads a Hugging Face checkpoint and runs it in PyTorch. The HTTP layer depends
> only on the `Engine` interface, so an alternative engine drops in unchanged.

### CLI

Commands are run as `uv run walnut <command>` (or `walnut <command>` inside an
activated venv):

- `uv run walnut serve MODEL [--host] [--port]` — serve `MODEL` (Hugging Face id
  or local path) at `/v1`. `--host`/`--port` also read
  `WALNUT_HOST`/`WALNUT_PORT`.
- `uv run walnut chat [--model] [--url] [--quick]` — chat with a served model.
  `--model` defaults to the first model the backend reports; `--url` selects the
  backend (default `http://127.0.0.1:8000/v1`); `--quick PROMPT` prints one
  completion and exits, otherwise an interactive REPL starts.

## Development

```bash
uv run ruff check      # lint
uv run ruff format     # format
uv run ty check        # type check
uv run pytest          # run tests
```

### Git hooks

Local hooks are managed with [prek](https://prek.j178.dev) (a drop-in
`pre-commit` replacement) and mirror the CI checks:

```bash
prek install           # enable the hook (once per clone)
prek run --all-files   # run ruff, ty, hadolint, and helm lint
```

Lint, type-check, tests, docs build, `helm lint`, and `hadolint` run in CI on
every push to `main` and all pull requests (see `.github/workflows/ci.yml`).
Docs are published to GitHub Pages on merge to `main`.

## Observability

The server emits structured JSON logs to stdout and Prometheus metrics at
`GET /metrics` (HTTP request metrics plus `walnut_chat_completions_total`). See
the [Observability guide](https://thomasbarrett.github.io/walnut/observability/).

## Container

```bash
docker build -t walnut .          # CPU image (default)
docker run --rm -p 8000:8000 walnut serve Qwen/Qwen3.5-0.8B
```

The image runs as a non-root user and reads `WALNUT_HOST` (default `0.0.0.0`)
and `WALNUT_PORT` (default `8000`); the model is passed as a `serve` argument.

torch is CPU-only by default. For a CUDA 13.0 image, pass the build arg:

```bash
docker build --build-arg TORCH_EXTRA=cu130 -t walnut:cuda .
```

Released images live at `ghcr.io/thomasbarrett/walnut`: CPU under the default
tags, CUDA under `-cuda` tags when enabled.

## Kubernetes (Helm)

```bash
helm install walnut charts/walnut --set model=Qwen/Qwen3.5-0.8B
```

The chart requires `model` and wires startup/readiness/liveness probes against
`/v1/models`. Enable Prometheus scraping with
`--set metrics.serviceMonitor.enabled=true`. See `charts/walnut/values.yaml` for
the full configuration surface (image, resources/GPU, autoscaling, ingress,
metrics).

## Releases

Releases are automated with
[release-please](https://github.com/googleapis/release-please) using
Conventional Commits. Merging the release PR bumps the version in
`pyproject.toml`, `walnut/__init__.py`, and `charts/walnut/Chart.yaml`, tags the
release, and publishes a container image to
`ghcr.io/thomasbarrett/walnut` tagged with the release version (see
`.github/workflows/release.yml`).

## License

Released under the [MIT License](LICENSE).
