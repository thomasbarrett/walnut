from walnut.engine import GenerationConfig, Message


def test_stream_defaults_to_single_chunk(stub_engine):
    messages = [Message(role="user", content="hi")]
    chunks = list(stub_engine.stream(messages, GenerationConfig()))
    assert chunks == [stub_engine.generate(messages, GenerationConfig())]
