import json
import logging

from walnut.telemetry import JsonFormatter


def _record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="uvicorn.error",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Uvicorn running on %s",
        args=("http://0.0.0.0:8000",),
        exc_info=None,
    )
    record.__dict__.update(extra)
    return record


def _format(**extra: object) -> dict:
    return json.loads(JsonFormatter().format(_record(**extra)))


def test_color_message_is_dropped():
    # uvicorn attaches an ANSI-colored copy of the message via ``extra=``.
    payload = _format(color_message="\x1b[1mrunning\x1b[0m")
    assert "color_message" not in payload
    assert payload["message"] == "Uvicorn running on http://0.0.0.0:8000"


def test_extra_context_is_serialized():
    payload = _format(request_id="abc", status_code=200)
    assert payload["request_id"] == "abc"
    assert payload["status_code"] == 200


def test_core_fields_present():
    payload = _format()
    assert payload["level"] == "INFO"
    assert payload["logger"] == "uvicorn.error"
    assert "timestamp" in payload
