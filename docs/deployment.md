# Deployment

## Container

```bash
docker build -t walnut .
docker run --rm -p 8000:8000 walnut serve Qwen/Qwen3.5-0.8B
```

The image runs as a non-root user and reads `WALNUT_HOST` (default `0.0.0.0`)
and `WALNUT_PORT` (default `8000`); the model is passed as a `serve` argument.

Released images are published to `ghcr.io/thomasbarrett/walnut`, tagged with
the release version (see [Development → Releases](development.md#releases)).

## Kubernetes (Helm)

```bash
helm install walnut charts/walnut --set model=Qwen/Qwen3.5-0.8B
```

The chart **requires** `model` and wires startup, readiness, and liveness
probes against `/v1/models`. See `charts/walnut/values.yaml` for the full
configuration surface (image, resources/GPU, autoscaling, ingress, metrics).

Prometheus metrics are served at `/metrics`; enable a `ServiceMonitor` with
`--set metrics.serviceMonitor.enabled=true`. See [Observability](observability.md).

### Linting the chart

```bash
helm lint charts/walnut --strict --values charts/walnut/ci/lint-values.yaml
helm template rel charts/walnut --values charts/walnut/ci/lint-values.yaml
```

Both run in CI (`helm-lint` job).
