# Observability

walnut exposes three signals: **structured logs**, **Prometheus metrics**, and
(later) **OpenTelemetry traces**.

## Logs

The server emits structured JSON logs to stdout — suitable for collection by
the platform (k8s, Loki, CloudWatch). Each request is logged with a generated
`request_id` (also returned in the `x-request-id` response header):

```json
{"timestamp": "2026-07-13T01:38:54Z", "level": "INFO", "logger": "walnut.server",
 "message": "http_request", "request_id": "11b351af…", "method": "POST",
 "path": "/v1/chat/completions", "status_code": 200, "duration_ms": 2.1}
```

uvicorn's own logs propagate through the same JSON handler, so all server
output is single-format. Logging is configured in `walnut.telemetry`; it is
applied by `walnut serve`, never as a side effect of importing the package.

## Metrics

Prometheus metrics are exposed at `GET /metrics` on the API port:

- **HTTP metrics** — request counts, latency histograms, and sizes, labelled by
  `handler`, `method`, and `status`.
- **`walnut_chat_completions_total`** — chat completion requests, labelled by
  `model` and `stream`.

```bash
curl http://127.0.0.1:8000/metrics
```

Model metrics (token throughput, time-to-first-token, queue depth) belong at the
`Engine` boundary and will land with the real engine. Today's metrics cover the
HTTP layer.

### Scraping in Kubernetes

The Helm chart can render a `ServiceMonitor` (Prometheus Operator) — off by
default so the chart doesn't require the CRD:

```yaml
metrics:
  serviceMonitor:
    enabled: true
    labels:
      release: kube-prometheus-stack   # match your Prometheus selector
    interval: 30s
```

## Traces

Not yet wired. OpenTelemetry traces are the next layer, once the real engine
produces spans worth tracing (prefill, decode, queue).
