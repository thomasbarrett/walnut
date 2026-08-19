from walnut.engine import GenerationConfig, Message


def test_stream_defaults_to_single_chunk(stub_engine):
    messages = [Message(role="user", content="hi")]
    chunks = list(stub_engine.stream(messages, GenerationConfig()))
    assert chunks == [stub_engine.generate(messages, GenerationConfig())]


def test_the_default_stream_reports_usage_once_it_is_exhausted(stub_engine):
    """Usage is only knowable at the end, which is where the schema puts it."""
    messages = [Message(role="user", content="hi there")]
    stream = stub_engine.stream(messages, GenerationConfig())
    assert stream.completion_tokens == 0
    list(stream)
    assert stream.usage.prompt_tokens == 2
    assert stream.usage.completion_tokens == 3  # "echo: hi there"
    assert stream.usage.total_tokens == 5
