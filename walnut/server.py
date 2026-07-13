"""OpenAI-compatible HTTP server, built on FastAPI.

Exposes the subset of the OpenAI API that ``walnut chat`` (and any other
OpenAI client) needs:

- ``GET  /v1/models``            — list the loaded model
- ``POST /v1/chat/completions``  — generate a completion (streaming optional)
- ``GET  /metrics``              — Prometheus metrics

All inference is delegated to an `Engine`, so this module has no knowledge of
PyTorch or model internals. Logging is set up in `walnut.telemetry`.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from .engine import Engine, GenerationConfig, Message
from .telemetry import configure_logging

logger = logging.getLogger("walnut.server")

# Walnut-specific metric, on top of the HTTP metrics the instrumentator adds.
CHAT_COMPLETIONS = Counter(
    "walnut_chat_completions_total",
    "Number of chat completion requests received.",
    ["model", "stream"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    stop: list[str] | str | None = None
    stream: bool = False


def _stop_list(stop: list[str] | str | None) -> list[str] | None:
    if stop is None:
        return None
    return [stop] if isinstance(stop, str) else stop


def create_app(engine: Engine) -> FastAPI:
    """Build a FastAPI app serving ``engine`` over the OpenAI-compatible API.

    A pure factory: it wires app-scoped instrumentation but does not configure
    global logging (that happens once in `serve`).
    """
    app = FastAPI(title="walnut", version="0.1.0")

    # HTTP request metrics + the /metrics endpoint (excluded from its own stats).
    Instrumentator(excluded_handlers=["/metrics"]).instrument(app).expose(
        app, endpoint="/metrics", include_in_schema=False
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        logger.info(
            "http_request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            },
        )
        return response

    @app.get("/v1/models")
    def list_models() -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "walnut",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        if not req.messages:
            raise HTTPException(status_code=400, detail="`messages` must not be empty")

        messages = [Message(role=m.role, content=m.content) for m in req.messages]
        config = GenerationConfig(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=_stop_list(req.stop),
        )
        completion_id = f"chatcmpl-{int(time.time() * 1000):x}"
        created = int(time.time())
        CHAT_COMPLETIONS.labels(
            model=engine.model_id, stream=str(req.stream).lower()
        ).inc()

        if req.stream:
            return StreamingResponse(
                _stream_chunks(engine, messages, config, completion_id, created),
                media_type="text/event-stream",
            )

        content = engine.generate(messages, config)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": engine.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def _stream_chunks(
    engine: Engine,
    messages: list[Message],
    config: GenerationConfig,
    completion_id: str,
    created: int,
) -> Iterator[str]:
    """Yield Server-Sent Events in the OpenAI streaming chunk format."""

    def event(delta: dict, finish_reason: str | None) -> str:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": engine.model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    yield event({"role": "assistant"}, None)
    for piece in engine.stream(messages, config):
        if piece:
            yield event({"content": piece}, None)
    yield event({}, "stop")
    yield "data: [DONE]\n\n"


def serve(engine: Engine, host: str, port: int) -> None:
    """Configure observability and run the server (blocking).

    This is the application entry point, so it always configures logging.
    uvicorn's access log is disabled in favor of the structured ``http_request``
    log emitted by the request middleware.
    """
    configure_logging()
    # log_config=None lets uvicorn's loggers propagate to our JSON root handler
    # instead of installing its own plain-text formatters.
    uvicorn.run(
        create_app(engine),
        host=host,
        port=port,
        access_log=False,
        log_config=None,
    )
