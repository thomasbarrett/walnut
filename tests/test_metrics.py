from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from tests.conftest import StubEngine
from walnut.engine import Stream
from walnut.server import create_app


class ChunkyEngine(StubEngine):
    """Streams a fixed number of pieces, so token timings have gaps to record."""

    def __init__(self, model_id: str = "test-model", pieces: int = 4) -> None:
        super().__init__(model_id)
        self.pieces = pieces

    def stream(self, messages, config) -> Stream:
        stream = Stream(prompt_tokens=1)

        def pieces():
            for index in range(self.pieces):
                stream.completion_tokens = index + 1
                yield f"tok{index} "

        stream.pieces = pieces()
        return stream


# StubEngine's model id, as both model attributes report it.
LABELS = {
    "gen_ai_operation_name": "chat",
    "gen_ai_provider_name": "walnut",
    "gen_ai_request_model": "test-model",
    "gen_ai_response_model": "test-model",
}

DURATION = "gen_ai_server_request_duration_seconds_count"
TTFT = "gen_ai_server_time_to_first_token_seconds_count"
TPOT = "gen_ai_server_time_per_output_token_seconds_count"


def _count(name: str, **extra: str) -> float:
    """Current value of a histogram's ``_count`` series.

    prometheus_client keeps one registry per process, so apps built across this
    suite share these series. Tests compare before/after, not presence.
    """
    return REGISTRY.get_sample_value(name, {**LABELS, **extra}) or 0.0


def _chat(client, **extra):
    return client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], **extra},
    )


def _stream(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as resp:
        list(resp.iter_lines())


def test_metrics_endpoint_exposes_prometheus(make_client):
    resp = make_client().get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


def test_chat_completion_records_request_duration(make_client):
    client = make_client()
    before = _count(DURATION, error_type="")
    _chat(client)
    assert _count(DURATION, error_type="") == before + 1


def test_convention_attributes_are_exposed(make_client):
    client = make_client()
    _chat(client)
    body = client.get("/metrics").text
    assert "gen_ai_server_request_duration_seconds_bucket" in body
    for key, value in LABELS.items():
        assert f'{key}="{value}"' in body
    assert 'error_type=""' in body


def test_rejected_request_is_not_timed(make_client):
    """A request that never reached the model isn't model-server latency."""
    client = make_client()
    before = _count(DURATION, error_type="")
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400
    assert _count(DURATION, error_type="") == before


def test_streaming_records_token_timings(make_client):
    client = make_client()
    before_ttft, before_duration = _count(TTFT), _count(DURATION, error_type="")
    _stream(client)
    assert _count(TTFT) == before_ttft + 1
    assert _count(DURATION, error_type="") == before_duration + 1


def test_every_token_after_the_first_is_timed():
    """First piece is TTFT; each later one is an inter-token gap."""
    client = TestClient(create_app(ChunkyEngine("test-model", pieces=4)))
    before_ttft, before_tpot = _count(TTFT), _count(TPOT)
    _stream(client)
    assert _count(TTFT) == before_ttft + 1
    assert _count(TPOT) == before_tpot + 3


def test_non_streaming_records_no_token_timings(make_client):
    """Token timing needs the streaming path; `generate` returns one string."""
    client = make_client()
    before = _count(TTFT)
    _chat(client)
    assert _count(TTFT) == before


def test_multiple_apps_do_not_double_register(make_client):
    # create_app is called per-test across the suite; ensure a second app in the
    # same process does not raise "Duplicated timeseries in CollectorRegistry".
    make_client()
    make_client()
