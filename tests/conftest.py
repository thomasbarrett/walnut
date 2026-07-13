import socket
import threading
import time

import httpx
import pytest
import uvicorn

from walnut.engine import EchoEngine
from walnut.server import create_app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def live_server():
    """Run the OpenAI-compatible server in a background thread.

    Yields the ``/v1`` base URL of a server backed by the EchoEngine.
    """
    port = _free_port()
    app = create_app(EchoEngine("test-model"))
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
