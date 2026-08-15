# Observability

walnut exposes **structured logs** and **Prometheus metrics** today, with
**OpenTelemetry traces** planned.

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

```bash
curl http://127.0.0.1:8000/metrics
```

**HTTP metrics** come from the instrumentator: request counts, latency
histograms, and sizes, labelled by `handler`, `method`, and `status`. They
describe the transport, not the model.

**Model metrics** follow the GenAI semantic conventions, below.

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

## GenAI semantic conventions

The [OpenTelemetry GenAI semantic
conventions](https://github.com/open-telemetry/semantic-conventions-genai) are a
standard vocabulary for LLM telemetry: agreed names for the metrics an inference
server publishes, the spans an LLM call produces, and the attributes on both. A
dashboard written against them works across implementations rather than against
one vendor's metric names.

walnut implements the model-server metrics, with the buckets the spec
prescribes — queries written against these conventions assume those boundaries:

| Metric | Records |
| --- | --- |
| `gen_ai_server_request_duration_seconds` | End-to-end request latency |
| `gen_ai_server_time_to_first_token_seconds` | Time until the first generated token |
| `gen_ai_server_time_per_output_token_seconds` | Gap between subsequent tokens |

### Names and attributes

The spec writes everything dotted: `gen_ai.server.request.duration`,
`gen_ai.operation.name`. Prometheus spells the same instruments with
underscores and a unit suffix, matching what OpenTelemetry's Prometheus
exporter emits — so that metric is `gen_ai_server_request_duration_seconds` on
`/metrics`.

Every series carries `gen_ai_operation_name` (always `chat`),
`gen_ai_provider_name`, `gen_ai_request_model`, and `gen_ai_response_model`.
Request duration adds `error_type`. The conventions drop that attribute on
success; a Prometheus histogram needs a fixed label set, so it is empty
instead.

Two values walnut had to choose:

- **`gen_ai_provider_name` is `walnut`.** The attribute tells consumers which
  provider's telemetry flavor to expect (`openai`, `anthropic`, …). walnut is
  OpenAI-compatible but not OpenAI, so `openai` would promise `openai.*`
  attributes it never emits. The spec allows a documented system-specific
  value.
- **Both model attributes report the loaded model**, not the request's `model`
  field. The engine ignores that field (see
  [HTTP API](http-api.md#walnut-specific-behavior)), and a client-supplied
  label value would let any caller mint unbounded series.

!!! note "Token timings cover streaming requests"

    Time to first token and time per output token come from the pieces
    `Engine.stream` yields, the only token-granular signal that interface
    exposes. Non-streaming requests record duration only, since `generate`
    returns one string. Timing both paths alike means instrumenting the engine.

### Not implemented

- **Spans** — the `chat {model}` span and its `gen_ai.request.*` /
  `gen_ai.response.*` attributes. See [Traces](#traces).
- **Token usage** — `gen_ai.usage.*` needs counts walnut doesn't produce; it
  also omits the `usage` block from chat completion responses.
- **Content capture** — prompts and completions as events. Opt-in, high volume,
  and a privacy decision.
- **Queue depth** — walnut serves one request at a time.
- **`gen_ai.client.*`** — those belong to clients calling walnut.

!!! warning "Stability"

    `gen_ai.*` names are at Development stability and may change before GA.
    `error.type`, `server.address`, and `server.port` are Stable, being core
    conventions rather than GenAI ones.

## Traces

Not yet wired. When they are, the request span follows the same [GenAI
conventions](#genai-semantic-conventions) as the metrics: named `chat {model}`,
carrying the request and response attributes, with child spans for prefill and
decode.

Metrics and traces answer different questions. A histogram says time to first
token regressed at p99; a span says this request took 812 ms, 780 of them in
decode. Both need the engine to expose phase boundaries — the same work the
token timings above are waiting on.
