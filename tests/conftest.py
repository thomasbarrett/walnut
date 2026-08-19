import socket
import threading
import time
from collections.abc import Callable

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from walnut.engine import Completion, Engine, GenerationConfig, Message, Usage
from walnut.server import create_app


class StubEngine(Engine):
    """Test double that echoes the last user turn without loading any weights.

    Lets the HTTP layer be exercised end to end with assertable responses.
    """

    def __init__(self, model_id: str = "test-model") -> None:
        self.model_id = model_id

    def complete(self, messages: list[Message], config: GenerationConfig) -> Completion:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        text = f"echo: {last_user}".strip()
        # No tokenizer here, so "tokens" are words — enough for the HTTP layer
        # to have counts to report.
        return Completion(
            text=text,
            usage=Usage(
                prompt_tokens=sum(len(m.content.split()) for m in messages),
                completion_tokens=len(text.split()),
            ),
        )


@pytest.fixture
def stub_engine() -> StubEngine:
    return StubEngine("test-model")


@pytest.fixture
def make_client() -> Callable[..., TestClient]:
    """Factory for a TestClient over a fresh app backed by a StubEngine."""

    def _make(model_id: str = "test-model") -> TestClient:
        return TestClient(create_app(StubEngine(model_id)))

    return _make


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def live_server():
    """Run the OpenAI-compatible server in a background thread.

    Yields the ``/v1`` base URL of a server backed by the StubEngine.
    """
    port = _free_port()
    app = create_app(StubEngine("test-model"))
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            httpx.get(f"{base}/v1/models", timeout=0.5)
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("server did not start in time")

    yield f"{base}/v1"

    server.should_exit = True
    thread.join(timeout=5)
