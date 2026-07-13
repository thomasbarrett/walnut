from walnut.engine import EchoEngine, GenerationConfig, Message, load_model


def test_load_model_returns_engine_with_model_id():
    engine = load_model("some/model")
    assert engine.model_id == "some/model"


def test_echo_engine_echoes_last_user_message():
    engine = EchoEngine("m")
    messages = [
        Message(role="system", content="be nice"),
        Message(role="user", content="hello there"),
    ]
    out = engine.generate(messages, GenerationConfig())
    assert "hello there" in out


def test_echo_engine_respects_max_tokens():
    engine = EchoEngine("m")
    messages = [Message(role="user", content="one two three four five")]
    out = engine.generate(messages, GenerationConfig(max_tokens=2))
    assert len(out.split()) == 2


def test_stream_defaults_to_single_chunk():
    engine = EchoEngine("m")
    messages = [Message(role="user", content="hi")]
    chunks = list(engine.stream(messages, GenerationConfig()))
    assert chunks == [engine.generate(messages, GenerationConfig())]
