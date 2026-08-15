"""OpenAI-compatible HTTP server, built on FastAPI.

Exposes the subset of the OpenAI API that ``walnut chat`` (and any other
OpenAI client) needs:

- ``GET  /v1/models``            — list the loaded model
- ``POST /v1/chat/completions``  — generate a completion (streaming optional)
- ``GET  /metrics``              — Prometheus metrics
- ``POST /start_profile``        — open a torch profiler window (opt-in)
- ``POST /stop_profile``         — close it and write a trace

All inference is delegated to an `Engine`, so this module has no knowledge of
PyTorch or model internals — the profiler arrives as a handle from `serve`
rather than being built here. Logging is set up in `walnut.telemetry`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from .engine import Engine, GenerationConfig, Message
from .telemetry import configure_logging

if TYPE_CHECKING:  # importing the profiler pulls in torch; keep it off the hot path
    from .profiler import ProfileArtifacts, TorchProfiler

logger = logging.getLogger("walnut.server")

# Model-server metrics from the OpenTelemetry GenAI semantic conventions:
# https://github.com/open-telemetry/semantic-conventions-genai
#
# The spec names these gen_ai.server.*; Prometheus spells the same instruments
# with underscores and a unit suffix. Buckets are the ones it prescribes, which
# queries written against the conventions assume.
_OPERATION = "chat"

# The attribute says which provider's telemetry flavor to expect. walnut is
# OpenAI-compatible but not OpenAI, so it uses its own name.
_PROVIDER = "walnut"

_LABELS = [
    "gen_ai_operation_name",
    "gen_ai_provider_name",
    "gen_ai_request_model",
    "gen_ai_response_model",
]

# fmt: off
_DURATION_BUCKETS = (
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64,
    1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92
)
_TTFT_BUCKETS = (
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,
    0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
)
_TPOT_BUCKETS = (
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2,
    0.3, 0.4, 0.5, 0.75, 1.0, 2.5
)
# fmt: on

REQUEST_DURATION = Histogram(
    "gen_ai_server_request_duration_seconds",
    "Generative AI server request duration such as time-to-last byte or last "
    "output token.",
    [*_LABELS, "error_type"],
    buckets=_DURATION_BUCKETS,
)

TIME_TO_FIRST_TOKEN = Histogram(
    "gen_ai_server_time_to_first_token_seconds",
    "Time to generate first token for successful responses.",
    _LABELS,
    buckets=_TTFT_BUCKETS,
)

TIME_PER_OUTPUT_TOKEN = Histogram(
    "gen_ai_server_time_per_output_token_seconds",
    "Time per output token generated after the first token for successful responses.",
    _LABELS,
    buckets=_TPOT_BUCKETS,
)


#: Profiling window if the caller doesn't say, and the ceiling it may ask for.
#: Bounded on purpose: an open-ended window grows the trace buffer until the
#: server dies. pprof, JFR, and Perfetto all take a duration for the same
#: reason.
DEFAULT_PROFILE_SECONDS = 30.0
MAX_PROFILE_SECONDS = 300.0


class ProfileRequest(BaseModel):
    duration_seconds: float = Field(
        DEFAULT_PROFILE_SECONDS, gt=0, le=MAX_PROFILE_SECONDS
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


def _labels(engine: Engine) -> dict[str, str]:
    """Attributes shared by every GenAI metric.

    Both model attributes name the loaded model: the engine ignores the
    request's ``model`` field, and a client-supplied label value would let any
    caller mint unbounded series.
    """
    return {
        "gen_ai_operation_name": _OPERATION,
        "gen_ai_provider_name": _PROVIDER,
        "gen_ai_request_model": engine.model_id,
        "gen_ai_response_model": engine.model_id,
    }


def _observe_duration(
    labels: dict[str, str], start: float, error_type: str = ""
) -> None:
    """Record one request against `REQUEST_DURATION`.

    The conventions omit ``error.type`` on success, but a Prometheus histogram
    needs a fixed label set, so absence is an empty value.
    """
    REQUEST_DURATION.labels(**labels, error_type=error_type).observe(
        time.perf_counter() - start
    )


def create_app(engine: Engine, profiler: TorchProfiler | None = None) -> FastAPI:
    """Build a FastAPI app serving ``engine`` over the OpenAI-compatible API.

    A pure factory: it wires app-scoped instrumentation but does not configure
    global logging (that happens once in `serve`).

    ``profiler`` enables ``/start_profile`` and ``/stop_profile``; without one
    those routes report 404.
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
        labels = _labels(engine)
        # Timed from here, past validation: a rejected request never ran the
        # model, and folding those in would skew the latency histogram low.
        start = time.perf_counter()

        if req.stream:
            return StreamingResponse(
                _stream_chunks(
                    engine, messages, config, completion_id, created, labels, start
                ),
                media_type="text/event-stream",
            )

        try:
            content = engine.generate(messages, config)
        except Exception as exc:
            _observe_duration(labels, start, type(exc).__qualname__)
            raise
        _observe_duration(labels, start)
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

    def _profiler() -> TorchProfiler:
        if profiler is None:
            raise HTTPException(
                status_code=404,
                detail="profiling is disabled; restart the server with "
                "WALNUT_TORCH_PROFILER_DIR set to an output directory",
            )
        return profiler

    # Both handlers are async on purpose. A sync handler runs in the threadpool,
    # which would start and stop the window on two different workers — kineto
    # segfaults on that. On the event loop they share one thread.
    deadline: asyncio.TimerHandle | None = None

    async def _end_window() -> ProfileArtifacts:
        """End the window here, then export off the event loop.

        `TorchProfiler.close` has to run on the thread that started the window,
        which is this one. Writing the trace can take seconds and is safe
        anywhere once the window is closed, so it goes to a worker rather than
        stalling every other request.
        """
        profiler = _profiler()
        profile = profiler.close()
        return await asyncio.get_running_loop().run_in_executor(
            None, profiler.write, profile
        )

    @app.post("/start_profile", include_in_schema=False)
    async def start_profile(req: ProfileRequest | None = None) -> dict:
        nonlocal deadline
        seconds = (req or ProfileRequest()).duration_seconds
        try:
            _profiler().start()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        async def expire() -> None:
            try:
                await _end_window()
            except RuntimeError:
                pass  # stopped by hand in the meantime
            except Exception:
                logger.exception("profile_autostop_failed")

        deadline = asyncio.get_running_loop().call_later(
            seconds, lambda: asyncio.ensure_future(expire())
        )
        return {"status": "profiling", "duration_seconds": seconds}

    @app.post("/stop_profile", include_in_schema=False)
    async def stop_profile() -> dict:
        nonlocal deadline
        if deadline is not None:
            deadline.cancel()
            deadline = None
        try:
            artifacts = await _end_window()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "stopped", **artifacts.as_dict()}

    return app


def _stream_chunks(
    engine: Engine,
    messages: list[Message],
    config: GenerationConfig,
    completion_id: str,
    created: int,
    labels: dict[str, str],
    start: float,
) -> Iterator[str]:
    """Yield Server-Sent Events in the OpenAI streaming chunk format.

    Token timings are taken here: the streamed pieces are the only
    token-granular signal `Engine` exposes, so only streaming requests get
    them.
    """

    def event(delta: dict, finish_reason: str | None) -> str:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": engine.model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    # The role delta carries no generated text, so it doesn't mark first token.
    yield event({"role": "assistant"}, None)
    previous: float | None = None
    error_type = ""
    try:
        for piece in engine.stream(messages, config):
            if not piece:
                continue
            now = time.perf_counter()
            if previous is None:
                TIME_TO_FIRST_TOKEN.labels(**labels).observe(now - start)
            else:
                TIME_PER_OUTPUT_TOKEN.labels(**labels).observe(now - previous)
            previous = now
            yield event({"content": piece}, None)
        yield event({}, "stop")
        yield "data: [DONE]\n\n"
    except Exception as exc:
        error_type = type(exc).__qualname__
        raise
    finally:
        _observe_duration(labels, start, error_type)


def serve(engine: Engine, host: str, port: int) -> None:
    """Configure observability and run the server (blocking).

    This is the application entry point, so it always configures logging.
    uvicorn's access log is disabled in favor of the structured ``http_request``
    log emitted by the request middleware. Profiling is wired here too, since
    reading the environment is an entry-point concern.
    """
    from .profiler import DIR_ENV, TorchProfiler

    configure_logging()
    profiler = TorchProfiler.from_env()
    if profiler is not None:
        logger.info("profiling_enabled", extra={"directory": str(profiler.directory)})
    else:
        logger.debug("profiling_disabled", extra={"env": DIR_ENV})
    # log_config=None lets uvicorn's loggers propagate to our JSON root handler
    # instead of installing its own plain-text formatters.
    uvicorn.run(
        create_app(engine, profiler=profiler),
        host=host,
        port=port,
        access_log=False,
        log_config=None,
    )
