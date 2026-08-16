# Observability

walnut exposes **structured logs**, **Prometheus metrics**, and an opt-in
**torch profiler**, with **OpenTelemetry traces** planned.

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

## Profiling

`torch.profiler` records a window and writes a Chrome trace, which opens at
[ui.perfetto.dev](https://ui.perfetto.dev/). walnut ships no viewer of its own.
Beside each trace is a `.summary.txt`, the profiler's `key_averages()` table,
for reading a run without a browser.

Profiling is expensive. On an RTX 5090 running Qwen3.5-0.8B it costs about 38%
of throughput (256 down to 185 tok/s) and grows the trace buffer by roughly
0.5 MB per token, which is only released when the window closes. So it is off
unless you turn it on, and every window is bounded.

### Phases

`walnut profile` records Python call frames, and kineto spells each one
`file(line): function`. Three functions exist to be found that way, so a trace
can be read per phase and per token:

| Frame ends with | Occurs | Covers |
| --- | --- | --- |
| `: _prefill` | once | The prompt forward pass, sampling, and the first token |
| `: capture` (in `graph.py`) | once, with `--cuda-graph` | Warming up and recording the decode graph |
| `: _decode_step` | once per generated token | Replay or forward, sampling, and the `.item()` sync |

Match on the name, not the line number — the line moves whenever the file
above it is edited:

```sql
select count(*) tokens, avg(dur)/1e6 avg_ms
from slice
where category = 'python_function' and name glob '*: _decode_step'
```

Each step's `.item()` — the sync where the CPU waits on the GPU — sits inside
the frame that produced the token. Outside it, a decode frame times only the
kernel launches and reads several times faster than the token really took.

There are no `record_function` scopes and so no `user_annotation` slices, as in
vLLM and SGLang. The two cannot coexist anyway: kineto interleaves
`python_function` slices with `user_annotation` ones in a way the trace
importer rejects, and it resolves that by dropping the annotations, silently.

### One-shot, from the CLI

`walnut profile` loads a model, warms it up, and profiles a single generation:

```bash
uv run walnut profile Qwen/Qwen3.5-0.8B --max-tokens 32 --output-dir ./profiles
```

```
trace:   profiles/walnut-20260815-182128-392836.trace.json.gz
summary: profiles/walnut-20260815-182128-392836.summary.txt
```

CUDA graphs are on by default here, as everywhere else: replaying decode from a
captured graph drops the per-token launch cost, so that is the configuration
worth measuring. `--no-cuda-graph` turns it off.

### On a running server

Set `WALNUT_TORCH_PROFILER_DIR` to enable `/start_profile` and `/stop_profile`,
then bracket the traffic you want:

```bash
WALNUT_TORCH_PROFILER_DIR=./profiles walnut serve Qwen/Qwen3.5-0.8B
```

```bash
curl -X POST localhost:8000/start_profile -d '{"duration_seconds": 30}'
# ...drive traffic...
curl -X POST localhost:8000/stop_profile
```

`stop_profile` returns the paths it wrote:

```json
{"status": "stopped",
 "trace": "profiles/walnut-20260815-182128-392836.trace.json.gz",
 "summary": "profiles/walnut-20260815-182128-392836.summary.txt"}
```

Windows are bounded. `duration_seconds` defaults to 30 and cannot exceed 300;
asking for more is a 422. When it expires the window closes itself and writes
the trace, so a caller that never sends `/stop_profile` cannot run the server
out of memory. `/stop_profile` ends a window early. pprof, JFR, and Perfetto
all bound captures the same way.

Without the environment variable both routes return 404. Starting twice, or
stopping when idle, is a 409. `summary` is `null` when the table could not be
built; the trace is written first and survives that.

!!! warning "These are admin endpoints"

    `/start_profile` is unauthenticated and sits on the API port. Anyone who
    can reach `/v1/chat/completions` can cost you a third of your throughput.
    Keep the port private, or leave `WALNUT_TORCH_PROFILER_DIR` unset in
    production. Kubernetes ships the same switch as `--profiling=false`, and
    the CIS benchmark requires it.

!!! note "Server profiles show kernels, not `aten::` ops"

    kineto records framework-level ops only on the thread that started the
    profile; GPU kernels are recorded from any thread. Requests run on
    threadpool workers, so a server profile gives the CUDA timeline without the
    `aten::` names above it. Use `walnut profile` for that attribution.

    The server also profiles without stacks: worker frames aren't recorded
    anyway, and collecting them inflates the trace and trips a parse failure in
    the averages table.

## Traces

Not yet wired. When they are, the request span follows the same [GenAI
conventions](#genai-semantic-conventions) as the metrics: named `chat {model}`,
carrying the request and response attributes, with child spans for prefill and
decode.

Metrics and traces answer different questions. A histogram says time to first
token regressed at p99; a span says this request took 812 ms, 780 of them in
decode. Both need the engine to expose phase boundaries — the same work the
token timings above are waiting on.
