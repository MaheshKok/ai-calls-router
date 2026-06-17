"""Centralized logging configuration and request correlation for the proxy.

This module owns the single place the proxy configures Python logging. It
binds a rotating file handler (the structured ``acr.log``) and a console
handler to the ``acr`` logger namespace only -- never the root logger -- so
third-party and pytest loggers are left untouched. A contextvar-backed
request id is stamped onto every record via a filter, letting one turn's log
lines be grepped out of interleaved concurrent traffic. setup_logging is
idempotent and re-reads the level env vars on each call.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING

from ai_calls_router._lib import config

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path
    from typing import TextIO

ACR_LOGGER_NAME = "acr"
"""Root of the proxy logger namespace; all module loggers are ``acr.*``."""

LOG_FORMAT = (
    "%(asctime)s %(levelname)-7s %(name)s %(filename)s:%(lineno)d [%(request_id)s] %(message)s"
)
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5

DEFAULT_FILE_LEVEL = logging.DEBUG
DEFAULT_CONSOLE_LEVEL = logging.INFO

LOG_DIR_MODE = 0o700
LOG_FILE_MODE = 0o600

_MANAGED_ATTR = "_acr_managed"
_setup_lock = threading.Lock()

_REQUEST_ID: ContextVar[str] = ContextVar("acr_request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Stamp the active request id onto every log record.

    The standard formatter cannot read a contextvar, so this filter copies
    the current correlation id (or the ``-`` sentinel outside any request)
    onto ``record.request_id`` for the format string to interpolate.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach ``request_id`` to the record; never drop the record.

        Args:
            record: The log record being processed.

        Returns:
            Always True -- this filter only enriches, it never suppresses.
        """
        record.request_id = _REQUEST_ID.get()
        return True


@contextmanager
def request_context(request_id: str | None = None) -> Generator[str, None, None]:
    """Bind a correlation id for the duration of the ``with`` block.

    Args:
        request_id: Explicit id to use; when None a fresh 12-char hex id is
            generated. Passing an upstream/trace id lets logs join across
            systems.

    Yields:
        The active correlation id, also retrievable via current_request_id.
    """
    rid = request_id or uuid.uuid4().hex[:12]
    token = _REQUEST_ID.set(rid)
    try:
        yield rid
    finally:
        _REQUEST_ID.reset(token)


def current_request_id() -> str:
    """Return the active correlation id, or ``-`` outside any request."""
    return _REQUEST_ID.get()


def _parse_level(value: str | None, default: int) -> int:
    """Resolve a level-name env value to a logging level int, fail-open.

    Args:
        value: Raw env value (e.g. "DEBUG", "warning") or None.
        default: Level to use when value is missing or unrecognized.

    Returns:
        The matching logging level, or default for blank/unknown names so a
        typo in the env never silences logging.
    """
    if not value or not value.strip():
        return default
    mapping = logging.getLevelNamesMapping()
    return mapping.get(value.strip().upper(), default)


def _ensure_log_dir(log_file: Path) -> None:
    """Create the log directory if absent, best-effort 0700 perms."""
    directory = log_file.parent
    directory.mkdir(parents=True, exist_ok=True)
    # Non-POSIX filesystem or insufficient privilege: directory exists, which
    # is what the handler needs; perms are a best-effort hardening.
    with contextlib.suppress(OSError):
        directory.chmod(LOG_DIR_MODE)


def _restrict_permissions(log_file: Path) -> None:
    """Tighten the log file to 0600, best-effort.

    The log can carry redacted request metadata; restricting it to the owner
    prevents other local users from reading it. Silently skipped where chmod
    is unsupported (the file still exists and logging proceeds).
    """
    with contextlib.suppress(OSError):
        log_file.chmod(LOG_FILE_MODE)


def _remove_managed_handlers(logger: logging.Logger) -> None:
    """Detach and close handlers this module previously attached.

    Makes setup_logging idempotent: repeated calls (e.g. test setup, daemon
    restart) replace our handlers instead of stacking duplicates, while
    leaving any externally-attached handlers in place.
    """
    for handler in list(logger.handlers):
        if getattr(handler, _MANAGED_ATTR, False):
            logger.removeHandler(handler)
            handler.close()


def _build_file_handler(level: int, request_filter: RequestIdFilter) -> RotatingFileHandler:
    """Build the rotating file handler targeting config.log_path()."""
    log_file = config.log_path()
    _ensure_log_dir(log_file)
    handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.addFilter(request_filter)
    setattr(handler, _MANAGED_ATTR, True)
    _restrict_permissions(log_file)
    return handler


def _build_console_handler(
    level: int, request_filter: RequestIdFilter
) -> logging.StreamHandler[TextIO]:
    """Build the stderr console handler for operator-visible lines."""
    handler: logging.StreamHandler[TextIO] = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.addFilter(request_filter)
    setattr(handler, _MANAGED_ATTR, True)
    return handler


def setup_logging() -> None:
    """Configure the ``acr`` logger namespace; idempotent and env-driven.

    Attaches a rotating file handler (config.log_path(), 10MB x5 backups) and
    a stderr console handler to the ``acr`` logger, each carrying the request
    id filter and a shared format. The file level is read from ``ACR_LOG_LEVEL``
    (default DEBUG) and the console level from ``ACR_CONSOLE_LOG_LEVEL``
    (default INFO); unrecognized values fail open to the defaults. Propagation
    to the root logger is disabled so the proxy never doubles into or hijacks
    third-party logging. Safe to call repeatedly: prior handlers this module
    installed are replaced, not duplicated.
    """
    file_level = _parse_level(os.environ.get("ACR_LOG_LEVEL"), DEFAULT_FILE_LEVEL)
    console_level = _parse_level(os.environ.get("ACR_CONSOLE_LOG_LEVEL"), DEFAULT_CONSOLE_LEVEL)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    request_filter = RequestIdFilter()

    with _setup_lock:
        logger = logging.getLogger(ACR_LOGGER_NAME)
        _remove_managed_handlers(logger)

        file_handler = _build_file_handler(file_level, request_filter)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = _build_console_handler(console_level, request_filter)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # The logger gate must sit at or below the most verbose handler, or it
        # would drop records before either handler sees them.
        logger.setLevel(min(file_level, console_level))
        logger.propagate = False
