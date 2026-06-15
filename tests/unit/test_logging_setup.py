"""Spec-derived tests for the logging_setup module.

These tests are written against the module's public contract -- the
request-id correlation helpers, the RequestIdFilter, and the idempotent
setup_logging configurator -- not its implementation. They assert
observable behavior: a fresh correlation id per context, a "-" sentinel
outside any context, handlers bound to the ``acr`` namespace (never root),
no handler duplication on repeated setup, records reaching the configured
log file with the request id stamped in, env-driven level filtering, and
0600 permissions on the log file.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from ai_calls_router._lib import logging_setup

ACR_LOGGER_NAME = "acr"


@pytest.fixture(autouse=True)
def _isolate_acr_logger() -> Iterator[None]:
    """Snapshot and restore the ``acr`` logger around each test.

    setup_logging mutates the process-global ``acr`` logger (handlers,
    level, propagate). Without isolation a configured handler from one test
    would write into another test's assertions, so the logger is stripped
    before the test and fully restored after.
    """
    logger = logging.getLogger(ACR_LOGGER_NAME)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_propagate = logger.propagate
    logger.handlers.clear()
    try:
        yield
    finally:
        for handler in list(logger.handlers):
            handler.close()
        logger.handlers.clear()
        logger.handlers.extend(saved_handlers)
        logger.setLevel(saved_level)
        logger.propagate = saved_propagate


@pytest.fixture
def acr_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ACR_HOME at a temp dir so config.log_path() is sandboxed."""
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    monkeypatch.delenv("ACR_LOG_LEVEL", raising=False)
    monkeypatch.delenv("ACR_CONSOLE_LOG_LEVEL", raising=False)
    return tmp_path


class TestRequestContext:
    """request_context / current_request_id correlation-id contract."""

    def test_generates_twelve_char_hex_id_when_none_given(self) -> None:
        with logging_setup.request_context() as rid:
            assert len(rid) == 12
            assert all(c in "0123456789abcdef" for c in rid)
            assert logging_setup.current_request_id() == rid

    def test_uses_provided_id_verbatim(self) -> None:
        with logging_setup.request_context("fixed-id-123") as rid:
            assert rid == "fixed-id-123"
            assert logging_setup.current_request_id() == "fixed-id-123"

    def test_distinct_ids_across_separate_contexts(self) -> None:
        with logging_setup.request_context() as first:
            pass
        with logging_setup.request_context() as second:
            pass
        assert first != second

    def test_restores_outer_id_on_nested_exit(self) -> None:
        with logging_setup.request_context("outer"):
            with logging_setup.request_context("inner"):
                assert logging_setup.current_request_id() == "inner"
            assert logging_setup.current_request_id() == "outer"

    def test_current_id_is_dash_outside_any_context(self) -> None:
        assert logging_setup.current_request_id() == "-"

    def test_id_reset_to_dash_after_context_exits(self) -> None:
        with logging_setup.request_context("temp"):
            pass
        assert logging_setup.current_request_id() == "-"


class TestRequestIdFilter:
    """RequestIdFilter stamps record.request_id and never drops records."""

    def _record(self) -> logging.LogRecord:
        return logging.LogRecord(
            name="acr.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="x",
            args=(),
            exc_info=None,
        )

    def test_returns_true_so_record_is_never_dropped(self) -> None:
        record = self._record()
        with logging_setup.request_context("abc"):
            assert logging_setup.RequestIdFilter().filter(record) is True

    def test_stamps_current_context_id(self) -> None:
        record = self._record()
        with logging_setup.request_context("req-9"):
            logging_setup.RequestIdFilter().filter(record)
        assert record.request_id == "req-9"

    def test_stamps_dash_outside_context(self) -> None:
        record = self._record()
        logging_setup.RequestIdFilter().filter(record)
        assert record.request_id == "-"


class TestSetupLogging:
    """setup_logging handler binding, idempotency, and file output."""

    def test_attaches_handlers_to_acr_logger(self, acr_home: Path) -> None:
        logging_setup.setup_logging()
        assert logging.getLogger(ACR_LOGGER_NAME).handlers

    def test_does_not_attach_handlers_to_root(self, acr_home: Path) -> None:
        before = len(logging.getLogger().handlers)
        logging_setup.setup_logging()
        assert len(logging.getLogger().handlers) == before

    def test_does_not_propagate_to_root(self, acr_home: Path) -> None:
        logging_setup.setup_logging()
        assert logging.getLogger(ACR_LOGGER_NAME).propagate is False

    def test_is_idempotent_no_duplicate_handlers(self, acr_home: Path) -> None:
        logging_setup.setup_logging()
        count_once = len(logging.getLogger(ACR_LOGGER_NAME).handlers)
        logging_setup.setup_logging()
        logging_setup.setup_logging()
        assert len(logging.getLogger(ACR_LOGGER_NAME).handlers) == count_once

    def test_writes_acr_records_to_log_file(self, acr_home: Path) -> None:
        logging_setup.setup_logging()
        logging.getLogger("acr.test").warning("canary-message-7788")
        log_file = acr_home / "acr.log"
        assert log_file.exists()
        assert "canary-message-7788" in log_file.read_text(encoding="utf-8")

    def test_includes_request_id_in_file_output(self, acr_home: Path) -> None:
        logging_setup.setup_logging()
        with logging_setup.request_context("trace-xyz"):
            logging.getLogger("acr.test").warning("with-correlation")
        contents = (acr_home / "acr.log").read_text(encoding="utf-8")
        assert "trace-xyz" in contents
        assert "with-correlation" in contents

    def test_dash_request_id_when_logged_outside_context(self, acr_home: Path) -> None:
        logging_setup.setup_logging()
        logging.getLogger("acr.test").warning("no-correlation-here")
        contents = (acr_home / "acr.log").read_text(encoding="utf-8")
        assert "[-]" in contents


class TestSetupLoggingLevels:
    """Env-driven level parsing for the file handler."""

    def test_acr_log_level_warning_suppresses_info_in_file(
        self, acr_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ACR_LOG_LEVEL", "WARNING")
        logging_setup.setup_logging()
        logging.getLogger("acr.test").info("info-should-be-filtered")
        logging.getLogger("acr.test").warning("warning-should-appear")
        contents = (acr_home / "acr.log").read_text(encoding="utf-8")
        assert "warning-should-appear" in contents
        assert "info-should-be-filtered" not in contents

    def test_acr_log_level_debug_captures_debug_in_file(
        self, acr_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ACR_LOG_LEVEL", "DEBUG")
        logging_setup.setup_logging()
        logging.getLogger("acr.test").debug("debug-breadcrumb-42")
        contents = (acr_home / "acr.log").read_text(encoding="utf-8")
        assert "debug-breadcrumb-42" in contents

    def test_invalid_level_falls_back_without_raising(
        self, acr_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ACR_LOG_LEVEL", "NOT_A_LEVEL")
        logging_setup.setup_logging()
        # Default file level is DEBUG; a warning must still be recorded.
        logging.getLogger("acr.test").warning("fallback-worked")
        contents = (acr_home / "acr.log").read_text(encoding="utf-8")
        assert "fallback-worked" in contents

    def test_lowercase_level_name_is_accepted(
        self, acr_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ACR_LOG_LEVEL", "warning")
        logging_setup.setup_logging()
        logging.getLogger("acr.test").info("lc-info-filtered")
        logging.getLogger("acr.test").warning("lc-warning-kept")
        contents = (acr_home / "acr.log").read_text(encoding="utf-8")
        assert "lc-warning-kept" in contents
        assert "lc-info-filtered" not in contents


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions only")
class TestSetupLoggingPermissions:
    """The log file must not be world/group readable."""

    def test_log_file_is_mode_0600(self, acr_home: Path) -> None:
        logging_setup.setup_logging()
        logging.getLogger("acr.test").warning("touch")
        mode = (acr_home / "acr.log").stat().st_mode & 0o777
        assert mode == 0o600
