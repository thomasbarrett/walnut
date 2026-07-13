from walnut.client import ChatClient


def test_list_and_default_model(live_server):
    with ChatClient(live_server) as client:
        assert client.list_models() == ["test-model"]
        assert client.default_model() == "test-model"


def test_chat(live_server):
    with ChatClient(live_server) as client:
        reply = client.chat("test-model", [{"role": "user", "content": "ping"}])
        assert "ping" in reply


def test_stream_chat(live_server):
    with ChatClient(live_server) as client:
        pieces = list(
            client.stream_chat("test-model", [{"role": "user", "content": "ping"}])
        )
        assert "ping" in "".join(pieces)
