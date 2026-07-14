def test_metrics_endpoint_exposes_prometheus(make_client):
    resp = make_client().get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


def test_chat_completion_increments_counter(make_client):
    client = make_client()
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    body = client.get("/metrics").text
    assert "walnut_chat_completions_total" in body


def test_multiple_apps_do_not_double_register(make_client):
    # create_app is called per-test across the suite; ensure a second app in the
    # same process does not raise "Duplicated timeseries in CollectorRegistry".
    make_client()
    make_client()
