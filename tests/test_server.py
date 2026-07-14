import json


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
