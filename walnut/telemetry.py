"""Observability wiring for walnut.

The signals follow the common inference-server split:

- **Logs** — structured JSON to stdout via `configure_logging`, suitable for
  collection by the platform (k8s, Loki, CloudWatch, ...). Configured once at
  the server entry point (`walnut.server.serve`).
- **Metrics** — Prometheus, scraped at ``/metrics`` (wired in `walnut.server`
  via ``prometheus-fastapi-instrumentator`` plus walnut-specific counters).
- **Traces** — not yet wired. OpenTelemetry traces are the intended next layer
  once the real engine produces meaningful spans (prefill/decode/queue).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

_LOGGING_CONFIGURED = False

# Standard LogRecord attributes; anything else on a record is treated as
# structured context passed via ``logger.info(..., extra={...})``.
_RESERVED_LOG_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}


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
        return json.dumps(payload, default=str)


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
