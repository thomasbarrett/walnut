"""Structured logging for the server.

Installs a JSON formatter on the root logger so walnut's logs and uvicorn's
both come out as single-line JSON on stdout, ready for a log collector to
parse. Set up once from `walnut.server.serve`.

Metrics live in `walnut.server` (Prometheus at ``/metrics``); traces aren't
wired yet.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

_LOGGING_CONFIGURED = False

# Attributes on a stdlib LogRecord are machinery, not context; everything else
# on a record is user-supplied ``extra=`` and gets serialized. ``color_message``
# is the one exception we add by hand: uvicorn injects an ANSI-colored copy of
# the message via ``extra=``, so it's noise rather than a built-in attribute.
_RESERVED_LOG_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"color_message"}


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON with any ``extra`` context."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Install a JSON stream handler on the root logger.

    Idempotent, and meant to be called once at the application entry point —
    not from `walnut.server.create_app`, which is a factory that tests and
    embedders call without wanting their logging config replaced.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    _LOGGING_CONFIGURED = True
