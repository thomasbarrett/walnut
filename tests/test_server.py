import json

from fastapi.testclient import TestClient

from walnut.engine import Completion, Engine, GenerationConfig, Usage
from walnut.scheduler import RequestError
from walnut.server import create_app


def test_list_models(make_client):
    resp = make_client().get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["test-model"]


def test_chat_completion(make_client):
    resp = make_client().post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "test-model"
    assert "ping" in body["choices"][0]["message"]["content"]
    assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_rejects_empty_messages(make_client):
    resp = make_client().post("/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 400


def test_chat_completion_streaming(make_client):
    with (
        make_client() as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "ping"}], "stream": True},
        ) as resp,
    ):
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line]

    payloads = [line[len("data: ") :] for line in lines if line.startswith("data: ")]
    assert payloads[-1] == "[DONE]"

    content = "".join(
        chunk
        for payload in payloads[:-1]
        for chunk in [json.loads(payload)["choices"][0]["delta"].get("content", "")]
    )
    assert "ping" in content


class _RejectingEngine(Engine):
    """Refuses everything, the way a request too long for the pool is."""

    model_id = "test-model"

    def complete(self, messages, config):
        raise RequestError("47 prompt tokens exceeds the 16-token context")

    def stream(self, messages, config):
        raise RequestError("47 prompt tokens exceeds the 16-token context")


def test_a_rejected_request_is_a_400_not_a_500():
    client = TestClient(create_app(_RejectingEngine()))
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400
    assert "16-token context" in response.json()["detail"]


def test_a_rejected_streaming_request_is_a_400_not_a_broken_stream():
    """The bug this guards: `_stream_chunks` is a generator, so an engine
    pulled inside it is pulled after the 200 has been committed — the client
    then sees a stream that stops with no status and no reason."""
    client = TestClient(create_app(_RejectingEngine()))
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert response.status_code == 400
    assert "16-token context" in response.json()["detail"]


def test_chat_completion_reports_usage(make_client):
    """The benchmark's token counts come from here, not from counting deltas."""
    resp = make_client().post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "ping"}]},
    )
    usage = resp.json()["usage"]
    assert usage["prompt_tokens"] == 1
    assert usage["completion_tokens"] == 2  # "echo: ping"
    assert usage["total_tokens"] == 3


def _sse(resp) -> list[dict]:
    payloads = [
        line[len("data: ") :] for line in resp.iter_lines() if line.startswith("data: ")
    ]
    assert payloads[-1] == "[DONE]"
    return [json.loads(p) for p in payloads[:-1]]


def test_streaming_omits_usage_unless_it_is_asked_for(make_client):
    with (
        make_client() as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "ping"}], "stream": True},
        ) as resp,
    ):
        chunks = _sse(resp)
    assert all("usage" not in chunk for chunk in chunks)


def test_streaming_usage_arrives_as_a_final_choiceless_chunk(make_client):
    """OpenAI's `stream_options.include_usage` shape, which clients look for:
    every content chunk carries `usage: null`, and one final chunk carries the
    counts with no choices."""
    with (
        make_client() as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "ping"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as resp,
    ):
        chunks = _sse(resp)

    assert all(chunk["usage"] is None for chunk in chunks[:-1])
    assert all(chunk["choices"] for chunk in chunks[:-1])
    final = chunks[-1]
    assert final["choices"] == []
    assert final["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }


class _RecordingEngine(Engine):
    """Keeps the config it was handed, so pass-through can be asserted."""

    model_id = "test-model"

    def __init__(self) -> None:
        self.config: GenerationConfig | None = None

    def complete(self, messages, config):
        self.config = config
        return Completion(text="ok", usage=Usage(1, 1))


def test_seed_and_ignore_eos_reach_the_engine():
    """Both are pass-through, but a benchmark that silently loses `ignore_eos`
    gets a different output length per request and never says so."""
    engine = _RecordingEngine()
    TestClient(create_app(engine)).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "ping"}],
            "seed": 7,
            "ignore_eos": True,
        },
    )
    assert engine.config is not None
    assert engine.config.seed == 7
    assert engine.config.ignore_eos is True
