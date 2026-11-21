"""Security-conscious logging for the HH agent.

A single redacting filter is attached to every handler installed by
:func:`configure_logging`.  It removes credentials, tokens, cookies,
Authorization headers and secret-bearing URL parameters from log records,
and replaces raw tracebacks with controlled exception identifiers.
"""

from __future__ import annotations

import copy
import logging
import re
import sys
from logging.handlers import RotatingFileHandler as _RotatingFileHandler
from pathlib import Path
from typing import Mapping

from security_utils import (
    ensure_validated_directory,
    open_validated_file,
    protect_log_file,
    validate_path_before_io,
)

REDACTED = "[REDACTED]"

_SECRET_KEYS = (
    "password",
    "passwd",
    "pwd",
    "proxy[_-]?authorization",
    "authorization",
    "access[_-]?token",
    "refresh[_-]?token",
    "id[_-]?token",
    "client[_-]?secret",
    # client_id must always be redacted: a fixed "authorization context" window
    # leaked long JSON records, multiline messages and reordered fields.
    "client[_-]?id",
    "tg[_-]?bot[_-]?token",
    "bot[_-]?token",
    "api[_-]?(?:key|secret|token)",
    "apikey",
    "private[_-]?key",
    "secret[_-]?key",
    "signing[_-]?(?:key|secret)",
    "webhook[_-]?(?:secret|token)",
    "auth[_-]?(?:code|token)",
    "oauth[_-]?token",
    "session[_-]?(?:id|identifier|token|key)",
    "sessionid",
    "credential(?:s)?",
    "secret",
    "token",
    "cookie",
    "sid",
)

_SECRET_ASSIGNMENT = re.compile(
    r'(?P<key>"?\b(?:' + "|".join(_SECRET_KEYS) + r')\b"?)'
    r"\s*(?P<separator>[:=])\s*"
    # ``]`` is allowed in a bare value so ``token=[REDACTED]leak`` is captured
    # as a whole instead of stopping at the placeholder's closing bracket.
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;}&)]+)",
    re.IGNORECASE,
)
# A sensitive header owns the whole remainder of its line.  Matching only the
# first token would leave additional Basic credentials or cookie pairs intact.
# The alternation is ordered longest-first so Proxy-Authorization and
# Set-Cookie are not partially matched by their shorter suffixes.
_SECRET_HEADER = re.compile(
    r"(?<![\w-])(?P<header>proxy-authorization|authorization|x-api-key|"
    r"x-auth-token|set-cookie|cookie)[ \t]*:[ \t]*(?P<value>[^\r\n]*)",
    re.IGNORECASE,
)
# A bearer value owns the rest of its line.  Otherwise a placeholder followed
# by a real credential would leave the credential visible.
_BEARER_TOKEN = re.compile(
    r"\bBearer[ \t]+(?P<value>[^\r\n]+)", re.IGNORECASE
)
_SECRET_URL_PARAMETER = re.compile(
    r"(?P<prefix>[?&](?:proxy_authorization|authorization|access_token|"
    r"refresh_token|id_token|client_secret|client_id|clientid|bot_token|tg_bot_token|api_key|"
    r"api_secret|api_token|apikey|password|passwd|secret|token|session_id|"
    r"sessionid|sid|auth|code|key|sig|"
    r"signature|ticket|state|credential|private_key)=)"
    r"(?P<value>[^&#\s]+)",
    re.IGNORECASE,
)

_SAFE_ERROR_IDS: Mapping[str, str] = {
    "BrowserLaunchError": "browser_launch_failed",
    "TimeoutError": "operation_timeout",
    "asyncio.TimeoutError": "operation_timeout",
    "ConnectionError": "connection_failed",
    "OSError": "filesystem_error",
    "FilesystemProtectionError": "filesystem_protection_failed",
    "ConfigError": "configuration_error",
    "HHAccountOwnershipError": "ownership_rejected",
    "TelegramAPIError": "telegram_api_error",
    "ValueError": "invalid_value",
}


def safe_exception_name(exc: BaseException | None) -> str:
    """Return only an exception type name, never its raw message."""
    return type(exc).__name__ if exc is not None else "None"


def safe_error_id(exc: BaseException | None) -> str:
    """Return a stable, secret-free identifier for an exception."""
    name = safe_exception_name(exc)
    return _SAFE_ERROR_IDS.get(name, "operation_failed")


def safe_exception_message(
    exc: BaseException | None,
    *,
    operation: str = "operation",
) -> str:
    """Build a diagnostic string that cannot carry exception payloads."""
    return (
        f"{operation}_failed "
        f"error_id={safe_error_id(exc)} "
        f"error_type={safe_exception_name(exc)}"
    )


def _already_redacted(value: str) -> bool:
    """Return True only when the whole value is the expected placeholder.

    A substring check would let an attacker keep a credential by prefixing it
    with ``[REDACTED]``.  Surrounding whitespace and a matching pair of quotes
    are accepted because those are part of the surrounding log format, not of
    the value itself.
    """
    candidate = value.strip()
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "\"'":
        candidate = candidate[1:-1].strip()
    return candidate == REDACTED


def _placeholder_tampered(value: str) -> bool:
    """A placeholder plus extra data is always treated as a real secret."""
    return REDACTED in value and not _already_redacted(value)


def _redacted_value(value: str) -> str:
    """Keep the original quoting style when replacing a quoted secret."""
    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        return f"{value[0]}{REDACTED}{value[0]}"
    return REDACTED


def redact_sensitive_text(value: object) -> str:
    """Redact known secret-bearing patterns in arbitrary log values."""
    text = str(value)

    def replace_header(match: re.Match[str]) -> str:
        if _already_redacted(match.group("value")):
            return match.group(0)
        return f"{match.group('header')}: {REDACTED}"

    def replace_assignment(match: re.Match[str]) -> str:
        if _already_redacted(match.group("value")):
            return match.group(0)
        # The whole value is consumed, so ``[REDACTED]tail`` cannot survive.
        return (
            f"{match.group('key')}{match.group('separator')}"
            f"{_redacted_value(match.group('value'))}"
        )

    def replace_bearer(match: re.Match[str]) -> str:
        if _already_redacted(match.group("value")):
            return match.group(0)
        return f"Bearer {REDACTED}"

    def replace_url_parameter(match: re.Match[str]) -> str:
        if _already_redacted(match.group("value")):
            return match.group(0)
        return f"{match.group('prefix')}{REDACTED}"

    text = _SECRET_HEADER.sub(replace_header, text)
    text = _SECRET_ASSIGNMENT.sub(replace_assignment, text)
    text = _BEARER_TOKEN.sub(replace_bearer, text)
    text = _SECRET_URL_PARAMETER.sub(replace_url_parameter, text)
    return text


def _logger_handle_with_record_replacement(
    logger: logging.Logger, record: logging.LogRecord
) -> None:
    """Python 3.11 backport of the 3.12 ``Logger.handle`` contract."""
    if logger.disabled:
        return
    maybe_record = logger.filter(record)
    if not maybe_record:
        return
    if isinstance(maybe_record, logging.LogRecord):
        record = maybe_record
    logger.callHandlers(record)


def _handler_handle_with_record_replacement(
    handler: logging.Handler, record: logging.LogRecord
) -> bool | logging.LogRecord:
    """Python 3.11 backport of the 3.12 ``Handler.handle`` contract."""
    result = handler.filter(record)
    if isinstance(result, logging.LogRecord):
        record = result
    if result:
        with handler.lock:
            handler.emit(record)
    return result


def _install_log_record_replacement_compat() -> None:
    """Make filter-returned records work on the project minimum Python 3.11.

    Python 3.12 added replacement of a ``LogRecord`` returned by a filter in
    both ``Logger.handle`` and ``Handler.handle``.  The project still supports
    3.11, so install a narrow compatibility shim there.  This preserves the
    no-mutation design instead of falling back to editing shared records.
    """
    if sys.version_info >= (3, 12):
        return
    if getattr(logging.Logger.handle, "_phase57_record_replacement", False):
        return
    logging.Logger.handle = _logger_handle_with_record_replacement  # type: ignore[method-assign]
    logging.Handler.handle = _handler_handle_with_record_replacement  # type: ignore[method-assign]
    logging.Logger.handle._phase57_record_replacement = True  # type: ignore[attr-defined]
    logging.Handler.handle._phase57_record_replacement = True  # type: ignore[attr-defined]


_install_log_record_replacement_compat()


class RedactingFilter(logging.Filter):
    """Return a redacted copy instead of mutating the shared LogRecord.

    Python 3.12 natively replaces a record returned by a filter in both
    ``Logger.handle`` and ``Handler.handle``.  On Python 3.11 the module
    installs a compatibility shim with the same semantics.  Each handler
    therefore receives its own redacted copy, while other handlers and filters
    still see the original ``msg``, ``args``, ``exc_info`` and ``stack_info``.
    """

    def filter(self, record: logging.LogRecord) -> logging.LogRecord | bool:
        if getattr(record, "_phase57_redacted", False):
            return True

        safe = copy.copy(record)
        message = redact_sensitive_text(safe.getMessage())
        if safe.exc_info:
            message = (
                f"{message} exception={safe_exception_name(safe.exc_info[1])}"
            )
        safe.msg = message
        safe.args = ()
        safe.message = message
        safe.exc_info = None
        safe.exc_text = None
        safe.stack_info = None
        safe._phase57_redacted = True
        return safe


class _ValidatedRotatingFileHandler(_RotatingFileHandler):
    """Rotating file handler whose initial open is handle-validated."""

    def _open(self):  # type: ignore[override]
        return open_validated_file(
            self.baseFilename,
            self.mode,
            kind="log file",
            encoding=self.encoding,
            errors=self.errors,
        )


# Keep the public module name used by configure_logging() and by the
# preflight-order regression test.
RotatingFileHandler = _ValidatedRotatingFileHandler


def configure_logging(log_path: Path) -> None:
    """Configure console + rotating file logging with one redacting filter."""
    log_path = Path(log_path)
    # Reject symlinks/junctions in every existing component before mkdir() or
    # the RotatingFileHandler can create a file outside the intended tree.
    validate_path_before_io(log_path, kind="log file")
    ensure_validated_directory(log_path.parent, kind="log directory")
    protect_log_file(log_path)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    redactor = RedactingFilter()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(redactor)

    rotating_file = RotatingFileHandler(
        log_path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    # The handler creates the file after the directory was protected.  Apply
    # protection again so the file itself has a protected DACL / 0600 mode.
    protect_log_file(log_path)
    rotating_file.setFormatter(formatter)
    rotating_file.addFilter(redactor)

    root = logging.getLogger()
    root.addFilter(redactor)
    logging.basicConfig(level=logging.INFO, handlers=[console, rotating_file], force=True)
