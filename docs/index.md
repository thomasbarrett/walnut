# walnut

An inference engine, built on PyTorch, with an OpenAI-compatible API.

walnut serves a model over the OpenAI chat API and ships a small CLI for
serving and chatting. The HTTP layer depends only on a narrow
[`Engine`](architecture.md) interface, so an alternative engine drops in
without touching the server.

!!! note "Pluggable engine"
    Inference runs through `TorchEngine` (`walnut/engine.py:load_model`), which
    loads a Hugging Face checkpoint and runs it in PyTorch. The server depends
    only on the `Engine` interface, so an alternative engine drops in unchanged.
    See [Architecture](architecture.md).

## Highlights

- **OpenAI-compatible** — `GET /v1/models` and `POST /v1/chat/completions`,
  with streaming.
- **CLI** — `walnut serve` and `walnut chat`.
- **Deployable** — `Dockerfile` (CPU / CUDA) and a Helm chart under `charts/walnut`.
- **Observability** — structured JSON logs and Prometheus metrics at `/metrics`.

## Where to next

<div class="grid cards" markdown>

- :material-rocket-launch: **[Getting started](getting-started.md)** — install and run.
- :material-console: **[CLI](cli.md)** — the `walnut` commands.
- :material-api: **[HTTP API](http-api.md)** — the OpenAI-compatible endpoints.
- :material-sitemap: **[Architecture](architecture.md)** — the engine seam.
- :material-server: **[Deployment](deployment.md)** — Docker and Kubernetes.
- :material-chart-line: **[Observability](observability.md)** — logs, metrics, traces.
- :material-book-open-variant: **[API reference](reference.md)** — generated from docstrings.

</div>
