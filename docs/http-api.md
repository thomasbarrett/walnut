# HTTP API

walnut serves the [OpenAI API](https://platform.openai.com/docs/api-reference)
under `/v1`, so any OpenAI-compatible client works against it. All inference is
delegated to an [`Engine`](reference.md#walnut.engine.Engine).

## Reference

The request/response schemas follow the OpenAI spec. Rather than restate them,
the running server generates the reference:

- **Interactive docs (Swagger UI):** `http://<host>:<port>/docs`
- **ReDoc:** `http://<host>:<port>/redoc`
- **Raw schema:** `http://<host>:<port>/openapi.json`

## Endpoints

walnut implements the subset needed by chat clients:

- `GET /v1/models` — list the loaded model.
- `POST /v1/chat/completions` — generate a completion (set `"stream": true`
  for a streamed response).

Two operational routes sit outside the OpenAI surface (and outside `/docs`):
`POST /start_profile` and `POST /stop_profile`, which return 404 unless the
server was started with `WALNUT_TORCH_PROFILER_DIR` set. See
[Observability](observability.md#profiling).

## walnut-specific behavior

Where walnut narrows or diverges from the spec:

- **One model.** The loaded model is always used; the request's `model` field
  is accepted but ignored. `GET /v1/models` reports that single model.
- **Empty `messages` → `400`.** A request with no messages is rejected.
- **`stop`** accepts either a single string or a list of strings.
- **Streaming** emits `chat.completion.chunk` events as `text/event-stream`,
  terminated by a `data: [DONE]` sentinel — the same shape the OpenAI SDK
  consumes.

## Quick check

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```
