"""Phase 5.7 credential and session-data security regression tests.

The 14 categories covered here are:

C1  Windows NTFS ACLs are restrictive, recursive and inherited.
C2  POSIX modes are 0700 for directories and 0600 for files.
C3  Path safety and fail-closed protection on errors/symlinks.
C4  Persistent browser profiles are protected and never rewritten.
C5  SQLite directory, database and journal files are protected.
C6  .env/profile.yaml secrets are protected and formats are preserved.
C7  Setup wizard keeps getpass and never prints a token prefix.
C8  Centralized logging redacts tokens, cookies and headers.
C9  Confirmed exception paths cannot leak raw messages.
C10 Logout clears IndexedDB, CacheStorage and Service Workers on HH origin.
C11 Partial cleanup failures are reported as failures.
C12 Cleanup never runs on an unrelated origin.
C13 Ownership, session locks and per-account isolation still hold.
C14 Logout and browser startup never delete profile data.
C15 SQLite path validation, symlink rejection and URI classification.
C16 False unregister/CacheStorage.delete results reach the cleanup report.
C17 Expanded secret redaction, idempotence and stack-trace suppression.
C18 Sensitive HTTP headers are redacted to the end of their line.
C19 Unsafe parent path components (symlinks and Windows reparse points).
C20 A placeholder inside a value can no longer hide the rest of the secret.
C21 Sensitive paths are validated before any read/write/mkdir operation.
C22 Full-audit regressions for placeholder, SQLite URI and identity edges.
"""

from __future__ import annotations

import asyncio
import config
import ctypes
import database as database_module
import logging
import logging_setup
import os
import re
import shutil
import subprocess
import sys
import tg_bot
from datetime import timedelta
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import hh_client as hh_client_module
import main
import security_utils
import setup_wizard
from approval import ApprovalGuard, ApplicationPermission
from browser_backend import (
    CloakBrowserBackend,
    PlaywrightBrowserBackend,
    create_browser_backend,
)
from config import ConfigError, load_settings
from database import Database, VacancyStatus
from hh_client import (
    HH_ACCOUNT_STATUS_AUTHORIZED,
    HH_ACCOUNT_STATUS_LOGGED_OUT,
    HHClient,
    LoginState,
    SessionCleanupReport,
    VacancySummary,
)
from logging_setup import (
    RedactingFilter,
    configure_logging,
    redact_sensitive_text,
    safe_exception_message,
)
from runtime_context import HHAccountOwnershipError
from security_utils import FilesystemProtectionError
from tests import test_approval as approval_tests
from tests.test_config import VALID_ENV, write_profile
from tests.test_hh_client import (
    NOW,
    LOGIN_MARKER,
    LOGIN_PAGE_URL,
    LogoutContext,
    LoginPage,
    VacancyPage,
    _approval_user_settings,
    _login_settings,
    _mark_account_authorized,
    _setup_hh_accounts,
)


SECRET = "SUPER-SECRET-TOKEN-DO-NOT-LOG"

_AUTHORIZED_PAGE = "https://hh.kz/applicant/resumes"
_SID_PATTERN = re.compile(r"S-1-\d+(?:-\d+)+")


# ---------------------------------------------------------------------------
# Cross-platform helpers
# ---------------------------------------------------------------------------


def _windows_sid_for_current_user() -> str:
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    return security_utils._current_user_sid(advapi32, kernel32)


def _expected_trustee_sids() -> set[str]:
    return {
        _windows_sid_for_current_user(),
        "S-1-5-18",  # SYSTEM
        "S-1-5-32-544",  # Administrators
    }


def _sddl_access(path: Path) -> str:
    """Read the DACL as SDDL without adding a PowerShell dependency to prod."""
    script = (
        "$p = $env:PHASE57_PATH; "
        "(Get-Acl -LiteralPath $p).GetSecurityDescriptorSddlForm('Access')"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        env={**os.environ, "PHASE57_PATH": str(path)},
    )
    return result.stdout.strip()


def _trustee_sids(path: Path) -> set[str]:
    sddl = _sddl_access(path)
    sids = set(_SID_PATTERN.findall(sddl))
    # PowerShell/SDDL uses short aliases for well-known trustees.  Normalize
    # them so the assertion compares identities, not display syntax.
    if re.search(r";;;SY\)", sddl):
        sids.add("S-1-5-18")
    if re.search(r";;;BA\)", sddl):
        sids.add("S-1-5-32-544")
    return sids


def _assert_restricted(path: Path) -> None:
    if os.name == "nt":
        assert _trustee_sids(path) == _expected_trustee_sids()
        # Files created inside a protected directory may carry inherited ACEs
        # instead of the protected flag; the trustee set is the security
        # invariant.  Directories must themselves be protected.
        if path.is_dir():
            assert "D:P" in _sddl_access(path)
        return
    expected = 0o700 if path.is_dir() else 0o600
    assert path.stat().st_mode & 0o777 == expected


def _restriction_snapshot(path: Path) -> object:
    if os.name == "nt":
        return _sddl_access(path)
    return path.stat().st_mode & 0o777


def _replace_with_link(link: Path, target: Path) -> None:
    """Replace ``link`` with a directory link to ``target`` after preflight.

    Windows policy in this environment forbids symlink creation, so a junction
    is used there; it still carries FILE_ATTRIBUTE_REPARSE_POINT and exercises
    the same handle-level rejection path.
    """
    if link.exists() or link.is_symlink():
        shutil.rmtree(link)
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            pytest.skip(
                f"junction creation unavailable: {result.stderr.strip()[:120]}"
            )
        return
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - depends on host policy
        pytest.skip(f"symlink creation unavailable: {type(exc).__name__}")


async def _logged_out_session() -> LoginState:
    return LoginState.NOT_AUTHORIZED


def _make_client(
    settings,
    database: Database,
    account,
    context,
    *,
    user_id: int = 42,
    lock: asyncio.Lock | None = None,
    checker=None,
    now=None,
) -> HHClient:
    effective_now = now or NOW
    user_settings = _approval_user_settings(settings)
    return HHClient(
        context,
        settings,
        database,
        ApprovalGuard(
            user_settings, database, now_factory=lambda: effective_now
        ),
        telegram_user_id=user_id,
        user_settings=user_settings,
        hh_account=account,
        sleep=lambda _seconds: asyncio.sleep(0),
        now_factory=lambda: effective_now,
        session_lock=lock,
        session_checker=checker or _logged_out_session,
    )


class PartialLogoutContext(LogoutContext):
    """Context where only IndexedDB cleanup fails."""

    async def _evaluate(self, script: str) -> dict[str, bool]:
        await super()._evaluate(script)
        return {
            "dom_storage": True,
            "indexeddb": False,
            "cachestorage": True,
            "service_worker": True,
        }


class FalseComponentLogoutContext(LogoutContext):
    """Context where one cleanup component reports an unsuccessful result."""

    def __init__(self, page, component: str):
        super().__init__(page)
        self.component = component

    async def _evaluate(self, script: str) -> dict[str, bool]:
        values = await super()._evaluate(script)
        values[self.component] = False
        return values


class GatedLogoutContext(LogoutContext):
    """Context that holds the first cleanup until a test releases it."""

    def __init__(self, page):
        super().__init__(page)
        self.started = 0
        self.gate = asyncio.Event()

    async def clear_cookies(self) -> None:
        self.started += 1
        if self.started == 1:
            await self.gate.wait()
        self.cookies_cleared = True


class SecretNewPageContext:
    """Context whose page creation embeds a secret in the raw exception."""

    async def new_page(self):
        raise RuntimeError(f"page creation failed token={SECRET}")


class _PageContext:
    def __init__(self, page):
        self.page = page

    async def new_page(self):
        return self.page


class SecretClosePage(VacancyPage):
    async def close(self) -> None:
        raise RuntimeError(f"page close failed cookie={SECRET}")


# ---------------------------------------------------------------------------
# C1 — Windows ACL hardening
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "nt", reason="Windows NTFS ACL test")
def test_c1_windows_acl_is_restrictive_recursive_and_inherited(
    tmp_path: Path,
) -> None:
    root = tmp_path / "hh-account-42"
    root.mkdir()
    child = root / "Default"
    child.mkdir()
    marker = child / "Cookies"
    marker.write_text("persisted-session", encoding="utf-8")

    security_utils.protect_browser_profile(root)

    expected = _expected_trustee_sids()
    for path in (root, child, marker):
        assert _trustee_sids(path) == expected
    directory_sddl = _sddl_access(root)
    assert "D:P" in directory_sddl
    assert "OICI" in directory_sddl  # object + container inheritance
    assert marker.read_text(encoding="utf-8") == "persisted-session"


# ---------------------------------------------------------------------------
# C2 — POSIX permission modes
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode test")
def test_c2_posix_modes_are_restrictive_and_recursive(tmp_path: Path) -> None:
    root = tmp_path / "hh-account-42"
    root.mkdir()
    child = root / "Default"
    child.mkdir()
    marker = child / "Cookies"
    marker.write_text("persisted-session", encoding="utf-8")
    root.chmod(0o777)
    child.chmod(0o777)
    marker.chmod(0o666)

    security_utils.protect_browser_profile(root)

    assert root.stat().st_mode & 0o777 == 0o700
    assert child.stat().st_mode & 0o777 == 0o700
    assert marker.stat().st_mode & 0o777 == 0o600
    assert marker.read_text(encoding="utf-8") == "persisted-session"


# ---------------------------------------------------------------------------
# C3 — Path safety and fail-closed behavior
# ---------------------------------------------------------------------------


def test_c3_protection_rejects_missing_wrong_type_and_symlinks(
    tmp_path: Path,
) -> None:
    with pytest.raises(FilesystemProtectionError):
        security_utils.protect_file(tmp_path / "missing-file")
    with pytest.raises(FilesystemProtectionError):
        security_utils.protect_directory(tmp_path / "missing-dir")

    regular = tmp_path / "regular"
    regular.write_text("data", encoding="utf-8")
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(FilesystemProtectionError):
        security_utils.protect_directory(regular)
    with pytest.raises(FilesystemProtectionError):
        security_utils.protect_file(directory)

    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "unrelated.txt"
    outside_file.write_text("do-not-touch", encoding="utf-8")
    profile = tmp_path / "profile"
    profile.mkdir()
    try:
        (profile / "link").symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - depends on Windows policy
        pytest.skip(f"symlink creation unavailable: {type(exc).__name__}")

    before = _restriction_snapshot(outside_file)
    security_utils.protect_browser_profile(profile)

    with pytest.raises(FilesystemProtectionError):
        security_utils.protect_directory(profile / "link")
    assert _restriction_snapshot(outside_file) == before
    assert outside_file.read_text(encoding="utf-8") == "do-not-touch"


def test_c3_protection_fails_closed_when_one_child_cannot_be_secured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "profile"
    root.mkdir()
    (root / "blocked").write_text("session", encoding="utf-8")

    if os.name == "nt":
        original = security_utils._apply_windows_acl

        def failing(path: Path, *, is_directory: bool) -> None:
            if path.name == "blocked":
                raise OSError("access denied")
            original(path, is_directory=is_directory)

        monkeypatch.setattr(security_utils, "_apply_windows_acl", failing)
        match = "Windows ACL update failed"
    else:
        original_posix = security_utils._apply_posix

        def failing_posix(path: Path, *, is_directory: bool) -> None:
            if path.name == "blocked":
                raise OSError("access denied")
            original_posix(path, is_directory=is_directory)

        monkeypatch.setattr(security_utils, "_apply_posix", failing_posix)
        match = "POSIX permission update failed"

    with pytest.raises(FilesystemProtectionError, match=match):
        security_utils.protect_browser_profile(root)


# ---------------------------------------------------------------------------
# C4 — Persistent browser profile protection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_type", [CloakBrowserBackend, PlaywrightBrowserBackend])
def test_c4_browser_start_protects_and_preserves_existing_profile(
    tmp_path: Path, backend_type: type
) -> None:
    profile = tmp_path / "profiles" / "hh-account-42"
    profile.mkdir(parents=True)
    marker = profile / "Local State"
    marker.write_text('{"persisted": true}', encoding="utf-8")
    context = object()

    async def launcher(profile_dir, *, headless: bool):
        assert Path(profile_dir) == profile
        return context

    backend = backend_type(profile, False, launcher=launcher)

    assert asyncio.run(backend.start()) is context
    assert profile.is_dir()
    assert marker.read_text(encoding="utf-8") == '{"persisted": true}'
    _assert_restricted(profile)
    _assert_restricted(marker)


def test_c4_explicit_account_profile_path_is_not_rewritten(tmp_path: Path) -> None:
    settings = _login_settings(tmp_path)
    custom_profile = tmp_path / "legacy" / "account-42-profile"

    backend = create_browser_backend(settings, profile_dir=custom_profile)

    assert backend.profile_dir == custom_profile
    assert backend.profile_dir != settings.browser_profile_dir / "hh-account-42"


# ---------------------------------------------------------------------------
# C5 — SQLite storage protection
# ---------------------------------------------------------------------------


def test_c5_database_init_protects_storage_and_preserves_rows(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "state" / "agent.db"
    database = Database(db_path)

    database.init()
    assert database.create_user(42, now=NOW)
    _assert_restricted(db_path.parent)
    _assert_restricted(db_path)
    assert database.get_user(42) is not None


def test_c5_database_helper_protects_existing_journal_siblings(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "agent.db"
    siblings = (
        db_path.parent / "agent.db-journal",
        db_path.parent / "agent.db-wal",
        db_path.parent / "agent.db-shm",
    )
    for sibling in siblings:
        sibling.write_text("sqlite-side-file", encoding="utf-8")

    security_utils.protect_database(db_path)

    for path in (db_path.parent, *siblings):
        _assert_restricted(path)
    for sibling in siblings:
        assert sibling.read_text(encoding="utf-8") == "sqlite-side-file"


# ---------------------------------------------------------------------------
# C6 — Configuration secret files
# ---------------------------------------------------------------------------


def test_c6_load_settings_protects_secret_files_without_changing_format(
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "".join(f"{key}={value}\n" for key, value in VALID_ENV.items()),
        encoding="utf-8",
    )
    profile_path = write_profile(tmp_path)
    original_env = env_path.read_text(encoding="utf-8")
    original_profile = profile_path.read_text(encoding="utf-8")

    settings = load_settings(
        env_path=env_path, profile_path=profile_path, environ={}
    )

    assert settings.tg_bot_token == VALID_ENV["TG_BOT_TOKEN"]
    assert settings.profile.candidate.name == "Test Candidate"
    assert env_path.read_text(encoding="utf-8") == original_env
    assert profile_path.read_text(encoding="utf-8") == original_profile
    _assert_restricted(env_path)
    _assert_restricted(profile_path)


def test_c6_config_errors_never_echo_secret_values(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "".join(f"{key}={value}\n" for key, value in VALID_ENV.items()),
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        f'candidate:\n  name: "{SECRET}"\n  desired_positions: [unclosed\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as raised:
        load_settings(env_path=env_path, profile_path=profile_path, environ={})

    assert SECRET not in str(raised.value)
    assert "cannot read profile:" in str(raised.value)
    _assert_restricted(profile_path)


# ---------------------------------------------------------------------------
# C7 — Setup wizard secret handling
# ---------------------------------------------------------------------------


def test_c7_wizard_uses_getpass_for_secret_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []
    visible_prompts: list[str] = []

    def fake_getpass(prompt: str) -> str:
        prompts.append(prompt)
        return SECRET

    def fake_input(prompt: str) -> str:
        visible_prompts.append(prompt)
        return "Candidate Name"

    monkeypatch.setattr(setup_wizard.getpass, "getpass", fake_getpass)
    monkeypatch.setattr(setup_wizard, "input", fake_input, raising=False)

    assert setup_wizard.ask("Token", secret=True) == SECRET
    assert setup_wizard.ask("Name") == "Candidate Name"
    assert len(prompts) == 1
    assert len(visible_prompts) == 1


def test_c7_wizard_never_prints_a_token_prefix(capsys: pytest.CaptureFixture[str]) -> None:
    setup_wizard.show_current_config(
        {"TG_BOT_TOKEN": SECRET, "TG_USER_ID": "42", "APP_MODE": "dry_run"},
        {},
    )

    output = capsys.readouterr().out
    assert SECRET not in output
    assert SECRET[:8] not in output
    assert "токен настроен" in output


# ---------------------------------------------------------------------------
# C8 — Centralized logging redaction
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_logging(monkeypatch: pytest.MonkeyPatch) -> list[logging.Handler]:
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_filters = list(root.filters)
    old_level = root.level
    installed: list[logging.Handler] = []

    def install_handlers(*, handlers=None, level=None, force: bool = False) -> None:
        installed.extend(handlers or [])
        root.handlers = list(handlers or [])
        if level is not None:
            root.setLevel(level)

    # Preserve pytest's capture handlers: the real configure_logging() uses
    # force=True, which would close them and corrupt later tests.
    monkeypatch.setattr(logging, "basicConfig", install_handlers)
    yield installed
    for handler in installed:
        handler.close()
    root.handlers = old_handlers
    root.filters = old_filters
    root.setLevel(old_level)


def test_c8_configure_logging_redacts_file_and_console(
    tmp_path: Path, isolated_logging: list[logging.Handler]
) -> None:
    log_path = tmp_path / "logs" / "agent.log"
    configure_logging(log_path)

    assert any(
        isinstance(item, RedactingFilter)
        for item in logging.getLogger().filters
    )
    assert isolated_logging
    assert all(
        any(isinstance(item, RedactingFilter) for item in handler.filters)
        for handler in isolated_logging
    )

    logging.getLogger("phase57").warning(
        "token=%s cookie=%s",
        SECRET,
        SECRET,
        exc_info=RuntimeError(f"Authorization: Bearer {SECRET}"),
    )
    for handler in isolated_logging:
        handler.flush()

    logged = log_path.read_text(encoding="utf-8")
    assert SECRET not in logged
    # Two secret-bearing values are redacted; the raw exception itself is
    # replaced by a controlled exception identifier rather than a placeholder.
    assert logged.count("[REDACTED]") >= 2
    assert "exception=RuntimeError" in logged
    assert "Traceback" not in logged
    _assert_restricted(log_path)
    _assert_restricted(log_path.parent)


def test_c8_redaction_covers_headers_tokens_and_url_parameters() -> None:
    # Header values own the rest of their line, so the independent patterns
    # are exercised on separate lines.
    text = "\n".join(
        (
            f"Authorization: Bearer {SECRET}",
            f"cookie={SECRET}",
            f"https://hh.kz/search?token={SECRET}&code=ok",
            f"password: {SECRET}",
        )
    )

    redacted = redact_sensitive_text(text)

    assert SECRET not in redacted
    assert redacted.count("[REDACTED]") >= 4


# ---------------------------------------------------------------------------
# C9 — Exception-path sanitization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_type", [CloakBrowserBackend, PlaywrightBrowserBackend])
def test_c9_browser_launch_error_never_copies_raw_exception(
    tmp_path: Path, backend_type: type
) -> None:
    async def broken_launcher(profile_dir, *, headless: bool):
        raise RuntimeError(f"binary blocked token={SECRET}")

    backend = backend_type(tmp_path / "profile", False, launcher=broken_launcher)

    with pytest.raises(Exception) as raised:
        asyncio.run(backend.start())

    message = str(raised.value)
    assert SECRET not in message
    assert "binary blocked" not in message
    assert "error_type=RuntimeError" in message


def test_c9_cli_startup_failure_is_sanitized(tmp_path: Path, monkeypatch, capsys) -> None:
    class FakeSettings:
        log_path = tmp_path / "agent.log"

    async def failing_run(settings, *, telegram_user_id=None):
        raise RuntimeError(f"browser failed token={SECRET}")

    monkeypatch.setattr(main, "load_settings", lambda *args, **kwargs: FakeSettings())
    monkeypatch.setattr(main, "configure_logging", lambda _path: None)
    monkeypatch.setattr(main, "run", failing_run)

    assert main.cli([]) == 1

    error_output = capsys.readouterr().err
    assert SECRET not in error_output
    assert "startup_failed" in error_output
    assert "error_type=RuntimeError" in error_output


def test_c9_cli_config_error_is_redacted(monkeypatch, capsys) -> None:
    def raise_config_error(*args, **kwargs):
        raise ConfigError(f"invalid setting token={SECRET}")

    monkeypatch.setattr(main, "load_settings", raise_config_error)

    assert main.cli([]) == 2

    error_output = capsys.readouterr().err
    assert SECRET not in error_output
    assert "token=[REDACTED]" in error_output


def test_c9_search_read_and_close_errors_are_sanitized(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    client = _make_client(settings, database, account_42, SecretNewPageContext())
    summary = VacancySummary("job-1", "Developer", "https://hh.kz/vacancy/1", "Python")

    with caplog.at_level(logging.WARNING):
        search_result = asyncio.run(
            client.search_vacancies("Python", (), ())
        )
        read_result = asyncio.run(client.read_vacancy(summary))

        close_page = SecretClosePage()
        close_client = _make_client(
            settings, database, account_42, _PageContext(close_page)
        )
        asyncio.run(close_client.read_vacancy(summary))

    assert SECRET not in search_result.error_reason
    assert "search_error:RuntimeError" in search_result.error_reason
    assert SECRET not in read_result.error
    assert "vacancy_read_failed" in read_result.error
    assert SECRET not in caplog.text
    assert "vacancy_page_close_failed" in caplog.text


def test_c9_application_error_text_is_sanitized(tmp_path: Path) -> None:
    app_settings = approval_tests.settings(
        tmp_path,
        APP_MODE="approval",
        ENABLE_REAL_APPLY="true",
        TG_USER_ID="42",
    )
    database = Database(app_settings.database_path)
    database.init()
    approval_tests.pending(database)
    token = database.approve("job-1", 42, 42, NOW + timedelta(minutes=1))
    assert token

    class SecretApplicationContext:
        async def new_page(self):
            raise RuntimeError(f"page creation failed token={SECRET}")

    user_settings = _approval_user_settings(app_settings)
    client = HHClient(
        SecretApplicationContext(),
        app_settings,
        database,
        ApprovalGuard(
            user_settings,
            database,
            now_factory=lambda: NOW + timedelta(minutes=2),
        ),
        telegram_user_id=42,
        user_settings=user_settings,
        session_checker=approval_tests.authorized_session,
        sleep=approval_tests.no_sleep,
    )

    sent = asyncio.run(
        client.submit_application(ApplicationPermission("job-1", token, 42))
    )

    assert not sent
    vacancy = database.get("job-1")
    assert vacancy is not None
    assert vacancy.status is VacancyStatus.APPLY_FAILED
    assert "application_submit_failed" in vacancy.error_text
    assert SECRET not in vacancy.error_text
    assert SECRET not in safe_exception_message(RuntimeError(f"token={SECRET}"))


# ---------------------------------------------------------------------------
# C10 — Full HH-origin session cleanup
# ---------------------------------------------------------------------------


def test_c10_logout_cleanup_covers_all_hh_origin_stores(tmp_path: Path) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database)

    page = LoginPage(
        url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,)
    )
    context = LogoutContext(page)
    context.forced_origin = "https://hh.kz/"
    client = _make_client(settings, database, account_42, context)

    report = asyncio.run(client._invalidate_session_unlocked())

    assert isinstance(report, SessionCleanupReport)
    assert report.ok is True
    assert report.failed == ()
    assert context.cookies_cleared is True
    assert "localStorage.clear();" in context.storage_script
    assert "sessionStorage.clear();" in context.storage_script
    assert "indexedDB.deleteDatabase" in context.storage_script
    assert "caches.keys" in context.storage_script
    assert "navigator.serviceWorker.getRegistrations" in context.storage_script
    assert context.goto_urls == ["https://hh.kz/"]
    assert context.legacy_storage_calls == 0

    result = asyncio.run(client.logout())
    assert result.ok is True
    assert result.state is LoginState.NOT_AUTHORIZED
    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_LOGGED_OUT


# ---------------------------------------------------------------------------
# C11 — Partial cleanup failure reporting
# ---------------------------------------------------------------------------


def test_c11_partial_component_failure_is_not_reported_as_success(
    tmp_path: Path,
) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database)

    page = LoginPage(url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,))
    context = PartialLogoutContext(page)
    context.forced_origin = "https://hh.kz/"
    client = _make_client(settings, database, account_42, context)

    report = asyncio.run(client._invalidate_session_unlocked())

    assert report.ok is False
    assert report.failed == ("indexeddb",)
    assert report.dom_storage is True
    assert report.cachestorage is True
    assert report.service_worker is True
    assert report.cookies is True

    result = asyncio.run(client.logout())

    assert result.ok is False
    assert result.state is LoginState.CHECK_FAILED
    assert result.reason == "invalidate_failed"
    stored = database.get_hh_account_model(42)
    assert stored is not None
    # Phase 5.6 intentionally keeps prior-authorization memory while recording
    # the safe technical error; the logout itself is still reported as failed.
    assert stored.status == HH_ACCOUNT_STATUS_AUTHORIZED
    assert stored.last_error
    assert "technical error" in stored.last_error.casefold()


# ---------------------------------------------------------------------------
# C12 — Origin enforcement
# ---------------------------------------------------------------------------


def test_c12_cleanup_never_touches_an_unrelated_origin(tmp_path: Path) -> None:
    from hh_client import _is_hh_origin

    assert _is_hh_origin("https://hh.kz/") is True
    assert _is_hh_origin("https://hh.kz.evil.example/") is False
    assert _is_hh_origin("http://hh.kz/") is False

    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database)

    page = LoginPage(url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,))
    context = LogoutContext(page)
    context.forced_origin = "https://evil.example/"
    client = _make_client(settings, database, account_42, context)

    report = asyncio.run(client._invalidate_session_unlocked())

    assert report.cookies is True
    assert report.ok is False
    assert report.dom_storage is False
    assert report.indexeddb is False
    assert report.cachestorage is False
    assert report.service_worker is False
    assert context.storage_script == ""
    assert context.goto_urls == ["https://hh.kz/"]

    result = asyncio.run(client.logout())
    assert result.ok is False
    assert result.reason == "invalidate_failed"


# ---------------------------------------------------------------------------
# C13 — Ownership, locks and per-account isolation
# ---------------------------------------------------------------------------


def test_c13_ownership_rejection_blocks_cross_account_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, account_43 = _setup_hh_accounts(database)
    user_settings = _approval_user_settings(settings)

    with pytest.raises(HHAccountOwnershipError):
        HHClient(
            SecretNewPageContext(),
            settings,
            database,
            ApprovalGuard(user_settings, database),
            telegram_user_id=43,
            user_settings=user_settings,
            hh_account=account_42,
        )

    page = LoginPage(url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,))
    context = LogoutContext(page)
    context.forced_origin = "https://hh.kz/"
    client = _make_client(settings, database, account_42, context)

    def reject_ownership(*args, **kwargs):
        raise HHAccountOwnershipError("ownership rejected")

    monkeypatch.setattr(
        hh_client_module, "verify_hh_account_ownership", reject_ownership
    )

    result = asyncio.run(client.logout())

    assert result.ok is False
    assert result.reason == "ownership"
    assert context.cookies_cleared is False
    assert context.storage_script == ""
    assert database.get_hh_account_by_id(account_43.id, 43) is not None


def test_c13_multi_account_logout_isolation(tmp_path: Path) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, account_43 = _setup_hh_accounts(database)
    _mark_account_authorized(database)
    assert database.confirm_hh_account_authorized(
        43,
        last_checked_at=NOW.isoformat(),
        last_error="",
        last_login_at=NOW.isoformat(),
    )

    page_a = LoginPage(url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,))
    context_a = LogoutContext(page_a)
    context_a.forced_origin = "https://hh.kz/"
    lock_a = asyncio.Lock()
    client_a = _make_client(
        settings, database, account_42, context_a, lock=lock_a
    )

    page_b = LoginPage(url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,))
    context_b = LogoutContext(page_b)
    context_b.forced_origin = "https://hh.kz/"
    lock_b = asyncio.Lock()
    client_b = _make_client(
        settings,
        database,
        account_43,
        context_b,
        user_id=43,
        lock=lock_b,
    )

    result = asyncio.run(client_a.logout())

    assert result.ok is True
    assert client_a.session_lock is lock_a
    assert client_b.session_lock is lock_b
    assert client_a.session_lock is not client_b.session_lock
    assert context_a.cookies_cleared is True
    assert context_b.cookies_cleared is False
    assert context_b.storage_script == ""
    stored_b = database.get_hh_account_by_id(account_43.id, 43)
    assert stored_b is not None
    assert stored_b.status == HH_ACCOUNT_STATUS_AUTHORIZED


def test_c13_session_lock_serialises_concurrent_logout_cleanup(
    tmp_path: Path,
) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database)

    page = LoginPage(url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,))
    context = GatedLogoutContext(page)
    context.forced_origin = "https://hh.kz/"
    lock = asyncio.Lock()
    client = _make_client(settings, database, account_42, context, lock=lock)

    async def scenario():
        first = asyncio.create_task(client.logout())
        second = asyncio.create_task(client.logout())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert context.started == 1
        context.gate.set()
        return await asyncio.gather(first, second)

    results = asyncio.run(scenario())

    assert context.started == 2
    assert all(result.ok for result in results)


# ---------------------------------------------------------------------------
# C14 — No destructive profile deletion
# ---------------------------------------------------------------------------


def test_c14_logout_preserves_persistent_profile_files(tmp_path: Path) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database)

    profile = tmp_path / "profiles" / "user_42"
    profile.mkdir(parents=True)
    default_dir = profile / "Default"
    default_dir.mkdir()
    markers = {
        profile / "Local State": '{"persisted": true}',
        default_dir / "Preferences": '{"session": "reused"}',
    }
    for path, content in markers.items():
        path.write_text(content, encoding="utf-8")

    page = LoginPage(url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,))
    context = LogoutContext(page)
    context.forced_origin = "https://hh.kz/"
    client = _make_client(settings, database, account_42, context)

    result = asyncio.run(client.logout())

    assert result.ok is True
    assert profile.is_dir()
    for path, content in markers.items():
        assert path.exists()
        assert path.read_text(encoding="utf-8") == content
    assert context.legacy_storage_calls == 0


# ---------------------------------------------------------------------------
# C15 — SQLite path validation and URI classification
# ---------------------------------------------------------------------------


def _make_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as exc:  # pragma: no cover - depends on Windows policy
        pytest.skip(f"symlink creation unavailable: {type(exc).__name__}")


def test_c15_existing_sqlite_symlink_is_rejected_before_connect(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.db"
    outside.write_text("unrelated-session-data", encoding="utf-8")
    before = _restriction_snapshot(outside)
    link = tmp_path / "agent.db"
    _make_symlink(link, outside)

    with pytest.raises(FilesystemProtectionError, match="symlink"):
        Database(link).init()

    assert link.is_symlink()
    assert outside.read_text(encoding="utf-8") == "unrelated-session-data"
    assert _restriction_snapshot(outside) == before


def test_c15_dangling_sqlite_symlink_is_rejected(tmp_path: Path) -> None:
    link = tmp_path / "agent.db"
    _make_symlink(link, tmp_path / "missing-target.db")

    with pytest.raises(FilesystemProtectionError, match="symlink"):
        Database(link).init()

    assert link.is_symlink()
    assert not link.exists()


def test_c15_ordinary_sqlite_path_is_validated_and_protected(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "state" / "agent.db"
    database = Database(db_path)

    database.init()
    assert database.create_user(42, now=NOW)

    assert db_path.is_file()
    _assert_restricted(db_path.parent)
    _assert_restricted(db_path)
    assert database.get_user(42) is not None


def test_c15_memory_databases_are_distinguished_from_file_uris() -> None:
    memory = security_utils.resolve_sqlite_target(":memory:")
    assert memory.in_memory is True
    assert memory.path is None
    assert memory.connect_name == ":memory:"
    assert memory.uri is False

    shared_memory = security_utils.resolve_sqlite_target(
        "file:shared?mode=memory&cache=shared"
    )
    assert shared_memory.in_memory is True
    assert shared_memory.path is None
    assert shared_memory.uri is True

    named_memory = security_utils.resolve_sqlite_target("file::memory:?cache=shared")
    assert named_memory.in_memory is True
    assert named_memory.path is None

    file_uri = security_utils.resolve_sqlite_target("file:agent.db?mode=rw")
    assert file_uri.in_memory is False
    assert file_uri.uri is True
    assert file_uri.path is not None
    assert file_uri.path.name == "agent.db"


def test_c15_memory_database_does_not_touch_the_filesystem(tmp_path: Path) -> None:
    database = Database(":memory:")

    # Each sqlite ':memory:' connection is independent by design; this test
    # only verifies that the application never protects or creates a file.
    database.init()
    with database._connect() as connection:
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    assert not (tmp_path / ":memory:").exists()


def test_c15_file_backed_sqlite_uri_is_protected(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "uri.db"
    uri = f"{db_path.as_uri()}?mode=rwc"
    target = security_utils.resolve_sqlite_target(uri)

    assert target.in_memory is False
    assert target.uri is True
    assert target.path == db_path.resolve()

    database = Database(uri)
    database.init()
    assert database.create_user(42, now=NOW)

    assert db_path.is_file()
    _assert_restricted(db_path.parent)
    _assert_restricted(db_path)
    assert database.get_user(42) is not None


# ---------------------------------------------------------------------------
# C16 — False cleanup operation results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("component", ["service_worker", "cachestorage"])
def test_c16_false_cleanup_result_is_reported_as_failed(
    tmp_path: Path, component: str
) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database)

    page = LoginPage(url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,))
    context = FalseComponentLogoutContext(page, component)
    context.forced_origin = "https://hh.kz/"
    client = _make_client(settings, database, account_42, context)

    report = asyncio.run(client._invalidate_session_unlocked())

    assert report.ok is False
    assert report.failed == (component,)
    assert getattr(report, component) is False
    assert "unregistered.every((ok) => ok === true)" in context.storage_script
    assert "deleted.every((ok) => ok === true)" in context.storage_script


# ---------------------------------------------------------------------------
# C17 — Expanded redaction
# ---------------------------------------------------------------------------


def test_c17_expanded_secret_patterns_are_idempotent() -> None:
    values = (
        "CLIENT-SECRET-1",
        "CLIENT-ID-SECRET",
        "TG-BOT-TOKEN-1",
        "BOT-TOKEN-1",
        "ACCESS-TOKEN-1",
        "REFRESH-TOKEN-1",
        "SESSION-ID-1",
        "API-KEY-1",
        "PRIVATE-KEY-1",
        "URL-SECRET-1",
        "OAUTH-CLIENT-1",
        "BEARER-SECRET-1",
    )
    text = " ".join(
        (
            f"client_secret={values[0]}",
            f"oauth request client_id={values[1]}",
            f"tg_bot_token: '{values[2]}'",
            f"bot_token=\"{values[3]}\"",
            f"access_token={values[4]}",
            f"refresh_token={values[5]}",
            f"session_id={values[6]}",
            f'{{"api_key": "{values[7]}"}}',
            f"private_key={values[8]}",
            f"https://api.example.com/token?client_secret={values[9]}",
            f"https://api.example.com/oauth?client_id={values[10]}&code=ok",
            f"Authorization: Bearer {values[11]}",
        )
    )

    redacted = redact_sensitive_text(text)

    for value in values:
        assert value not in redacted
    assert redact_sensitive_text(redacted) == redacted


def test_c17_safe_diagnostic_fields_are_preserved() -> None:
    text = "request_id=req-42 correlation_id=corr-7 status=ok client_id=public-analytics"
    expected = (
        "request_id=req-42 correlation_id=corr-7 status=ok "
        "client_id=[REDACTED]"
    )

    assert redact_sensitive_text(text) == expected


def test_c17_exception_and_stack_trace_cannot_leak_secrets(
    tmp_path: Path, isolated_logging: list[logging.Handler]
) -> None:
    log_path = tmp_path / "logs" / "agent.log"
    configure_logging(log_path)

    try:
        raise RuntimeError(f"client_secret={SECRET} session_id={SECRET}")
    except RuntimeError:
        logging.getLogger("phase57").error(
            "oauth_failure", exc_info=True, stack_info=True
        )
    for handler in isolated_logging:
        handler.flush()

    logged = log_path.read_text(encoding="utf-8")
    assert SECRET not in logged
    assert "Traceback" not in logged
    assert "exception=RuntimeError" in logged


# ---------------------------------------------------------------------------
# C18 — Sensitive HTTP headers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "secrets"),
    [
        (
            "Authorization: Basic dXNlcjpzZWNyZXQ=",
            ("dXNlcjpzZWNyZXQ=",),
        ),
        (
            'Authorization: Digest username="u", realm="r", '
            'nonce="NONCE-SECRET", response="RESPONSE-SECRET"',
            ("NONCE-SECRET", "RESPONSE-SECRET"),
        ),
        (
            "Proxy-Authorization: Basic PROXY-SECRET",
            ("PROXY-SECRET",),
        ),
        (
            "Cookie: session=COOKIE-1; other=COOKIE-2",
            ("COOKIE-1", "COOKIE-2"),
        ),
        (
            "Set-Cookie: session=SET-1; Path=/; HttpOnly",
            ("SET-1",),
        ),
        (
            "X-API-Key: API-KEY-HEADER",
            ("API-KEY-HEADER",),
        ),
        (
            "X-Auth-Token: AUTH-TOKEN-HEADER",
            ("AUTH-TOKEN-HEADER",),
        ),
    ],
)
def test_c18_sensitive_header_value_is_redacted_to_end_of_line(
    line: str, secrets: tuple[str, ...]
) -> None:
    redacted = redact_sensitive_text(line)

    header_name = line.split(":", 1)[0]
    assert redacted.startswith(f"{header_name}:")
    for secret in secrets:
        assert secret not in redacted
    assert "[REDACTED]" in redacted
    assert redact_sensitive_text(redacted) == redacted


def test_c18_multiline_headers_do_not_swallow_safe_lines() -> None:
    safe_line = "status=ok request_id=req-1 correlation_id=corr-7"
    lines = (
        "Authorization: Basic BASIC-MULTI",
        "Cookie: a=A-1; b=B-2",
        "Proxy-Authorization: Basic PROXY-MULTI",
        safe_line,
        "Set-Cookie: c=C-3; HttpOnly",
        "X-API-Key: KEY-MULTI",
        "X-Auth-Token: TOKEN-MULTI",
    )
    secrets = (
        "BASIC-MULTI",
        "A-1",
        "B-2",
        "PROXY-MULTI",
        "C-3",
        "KEY-MULTI",
        "TOKEN-MULTI",
    )

    redacted = redact_sensitive_text("\n".join(lines))

    for secret in secrets:
        assert secret not in redacted
    assert safe_line in redacted
    assert redacted.count("[REDACTED]") == len(lines) - 1
    assert redact_sensitive_text(redacted) == redacted


# ---------------------------------------------------------------------------
# C19 — Unsafe parent path components
# ---------------------------------------------------------------------------


def _symlinked_parent(tmp_path: Path) -> tuple[Path, Path]:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked"
    _make_symlink(link, outside)
    return outside, link


def test_c19_regular_file_inside_symlink_parent_is_rejected(tmp_path: Path) -> None:
    outside, link = _symlinked_parent(tmp_path)
    target = outside / "agent.db"
    target.write_text("session-data", encoding="utf-8")
    before = _restriction_snapshot(target)

    with pytest.raises(FilesystemProtectionError, match="unsafe path component"):
        security_utils.protect_file(link / "agent.db")

    assert target.read_text(encoding="utf-8") == "session-data"
    assert _restriction_snapshot(target) == before


def test_c19_directory_inside_symlink_parent_is_rejected(tmp_path: Path) -> None:
    outside, link = _symlinked_parent(tmp_path)
    target_dir = outside / "profile"
    target_dir.mkdir()
    marker = target_dir / "Preferences"
    marker.write_text('{"persisted": true}', encoding="utf-8")
    before_dir = _restriction_snapshot(target_dir)
    before_marker = _restriction_snapshot(marker)

    with pytest.raises(FilesystemProtectionError, match="unsafe path component"):
        security_utils.protect_directory(link / "profile")

    assert marker.read_text(encoding="utf-8") == '{"persisted": true}'
    assert _restriction_snapshot(target_dir) == before_dir
    assert _restriction_snapshot(marker) == before_marker


def test_c19_config_file_inside_symlink_parent_is_rejected(tmp_path: Path) -> None:
    outside, link = _symlinked_parent(tmp_path)
    target = outside / ".env"
    target.write_text(f"TG_BOT_TOKEN={SECRET}\n", encoding="utf-8")
    before = _restriction_snapshot(target)

    with pytest.raises(FilesystemProtectionError, match="unsafe path component"):
        security_utils.protect_config_file(link / ".env")

    assert target.read_text(encoding="utf-8") == f"TG_BOT_TOKEN={SECRET}\n"
    assert _restriction_snapshot(target) == before


def test_c19_browser_profile_inside_symlink_parent_is_rejected(
    tmp_path: Path,
) -> None:
    outside, link = _symlinked_parent(tmp_path)
    profile = outside / "hh-account-42"
    profile.mkdir()
    marker = profile / "Local State"
    marker.write_text('{"persisted": true}', encoding="utf-8")
    before = _restriction_snapshot(marker)

    with pytest.raises(FilesystemProtectionError, match="unsafe path component"):
        security_utils.protect_browser_profile(link / "hh-account-42")

    assert marker.read_text(encoding="utf-8") == '{"persisted": true}'
    assert _restriction_snapshot(marker) == before


def test_c19_log_and_database_paths_inside_symlink_parent_are_rejected(
    tmp_path: Path,
) -> None:
    outside, link = _symlinked_parent(tmp_path)
    log_file = outside / "agent.log"
    log_file.write_text("diagnostic", encoding="utf-8")
    database_file = outside / "agent.db"
    database_file.write_text("sqlite-data", encoding="utf-8")
    before_log = _restriction_snapshot(log_file)
    before_db = _restriction_snapshot(database_file)

    with pytest.raises(FilesystemProtectionError, match="unsafe path component"):
        security_utils.protect_log_file(link / "agent.log")
    with pytest.raises(FilesystemProtectionError, match="unsafe path component"):
        security_utils.protect_database(link / "agent.db")

    assert log_file.read_text(encoding="utf-8") == "diagnostic"
    assert database_file.read_text(encoding="utf-8") == "sqlite-data"
    assert _restriction_snapshot(log_file) == before_log
    assert _restriction_snapshot(database_file) == before_db


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_c19_windows_junction_parent_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "agent.db"
    marker.write_text("session-data", encoding="utf-8")
    before = _restriction_snapshot(marker)
    junction = tmp_path / "junction"

    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"junction creation unavailable: {result.stderr.strip()[:120]}")

    try:
        with pytest.raises(FilesystemProtectionError, match="unsafe path component"):
            security_utils.protect_file(junction / "agent.db")
        assert marker.read_text(encoding="utf-8") == "session-data"
        assert _restriction_snapshot(marker) == before
    finally:
        if junction.exists():
            junction.rmdir()


# ---------------------------------------------------------------------------
# C20 — Placeholder-prefix bypass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("[REDACTED]", True),
        ("  [REDACTED]  ", True),
        ('"[REDACTED]"', True),
        ("'[REDACTED]'", True),
        ("[REDACTED] ACTUAL-SECRET", False),
        ("[REDACTED]ACTUAL-SECRET", False),
        ("ACTUAL-SECRET [REDACTED]", False),
        ("REDACTED", False),
        ("[REDACTED", False),
        ("", False),
    ],
)
def test_c20_already_redacted_requires_the_whole_placeholder(
    value: str, expected: bool
) -> None:
    assert logging_setup._already_redacted(value) is expected


@pytest.mark.parametrize(
    ("text", "secrets"),
    [
        ("Authorization: [REDACTED] ACTUAL-SECRET", ("ACTUAL-SECRET",)),
        ("Cookie: [REDACTED] ACTUAL-SECRET", ("ACTUAL-SECRET",)),
        ("Proxy-Authorization: [REDACTED] ACTUAL-PROXY", ("ACTUAL-PROXY",)),
        ("Set-Cookie: [REDACTED] ACTUAL-SET", ("ACTUAL-SET",)),
        ("X-API-Key: [REDACTED] ACTUAL-KEY", ("ACTUAL-KEY",)),
        ("X-Auth-Token: [REDACTED] ACTUAL-AUTH", ("ACTUAL-AUTH",)),
        ("token=[REDACTED]ACTUAL-SECRET", ("ACTUAL-SECRET",)),
        ("Bearer [REDACTED] ACTUAL-SECRET", ("ACTUAL-SECRET",)),
        (
            "https://hh.kz/search?token=[REDACTED]ACTUAL-URL&code=ok",
            ("ACTUAL-URL",),
        ),
        ('{"api_key": "[REDACTED]ACTUAL-JSON"}', ("ACTUAL-JSON",)),
        ("oauth client_id=[REDACTED]ACTUAL-CLIENT", ("ACTUAL-CLIENT",)),
        (
            "https://hh.kz/oauth?client_id=[REDACTED]ACTUAL-CID&code=ok",
            ("ACTUAL-CID",),
        ),
    ],
)
def test_c20_placeholder_prefix_cannot_hide_the_remaining_secret(
    text: str, secrets: tuple[str, ...]
) -> None:
    redacted = redact_sensitive_text(text)

    for secret in secrets:
        assert secret not in redacted
    assert redacted.count("[REDACTED]") >= 1
    assert redact_sensitive_text(redacted) == redacted


@pytest.mark.parametrize(
    "text",
    [
        "Authorization: [REDACTED]",
        "Cookie: [REDACTED]",
        "token=[REDACTED]",
        "Bearer [REDACTED]",
        "https://hh.kz/search?token=[REDACTED]",
        '{"api_key": "[REDACTED]"}',
        "oauth client_id=[REDACTED]",
        "Authorization:   [REDACTED]  \nBearer  [REDACTED]  ",
    ],
)
def test_c20_correctly_redacted_values_stay_unchanged(text: str) -> None:
    redacted = redact_sensitive_text(text)

    assert redacted == text
    assert redact_sensitive_text(redacted) == redacted


# ---------------------------------------------------------------------------
# C21 — Preflight validation before filesystem operations
# ---------------------------------------------------------------------------


def test_c21_preflight_allows_missing_target_without_modifying_anything(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "config"
    parent.mkdir()
    target = parent / ".env"
    before_parent = _restriction_snapshot(parent)

    returned = security_utils.validate_path_before_io(
        target, kind="configuration file"
    )

    assert returned == target
    assert not target.exists()
    assert _restriction_snapshot(parent) == before_parent


def test_c21_env_write_is_rejected_before_touching_external_file(
    tmp_path: Path, monkeypatch
) -> None:
    outside, link = _symlinked_parent(tmp_path)
    target = outside / ".env"
    original = "TG_BOT_TOKEN=OLD-SECRET\n"
    target.write_text(original, encoding="utf-8")
    before = _restriction_snapshot(target)
    monkeypatch.setattr(setup_wizard, "ENV_PATH", link / ".env")

    with pytest.raises(FilesystemProtectionError, match="unsafe path component"):
        setup_wizard._write_env("TG_BOT_TOKEN=NEW-SECRET\n")

    assert target.read_text(encoding="utf-8") == original
    assert _restriction_snapshot(target) == before


def test_c21_profile_write_is_rejected_before_touching_external_file(
    tmp_path: Path, monkeypatch
) -> None:
    outside, link = _symlinked_parent(tmp_path)
    target = outside / "profile.yaml"
    original = "candidate:\n  name: OLD-CANDIDATE\n"
    target.write_text(original, encoding="utf-8")
    before = _restriction_snapshot(target)
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    monkeypatch.setattr(setup_wizard, "ENV_PATH", safe_dir / ".env")
    monkeypatch.setattr(setup_wizard, "PROFILE_PATH", link / "profile.yaml")

    with pytest.raises(FilesystemProtectionError, match="unsafe path component"):
        setup_wizard.write_files("# env\n", "candidate:\n  name: NEW-CANDIDATE\n")

    assert target.read_text(encoding="utf-8") == original
    assert _restriction_snapshot(target) == before


def test_c21_wizard_edit_profile_save_is_rejected_before_write(
    tmp_path: Path, monkeypatch
) -> None:
    outside, link = _symlinked_parent(tmp_path)
    target = outside / "profile.yaml"
    original = "candidate:\n  name: OLD-CANDIDATE\n"
    target.write_text(original, encoding="utf-8")
    before = _restriction_snapshot(target)
    monkeypatch.setattr(setup_wizard, "PROFILE_PATH", link / "profile.yaml")
    monkeypatch.setattr(setup_wizard, "show_current_config", lambda *a, **k: None)
    monkeypatch.setattr(setup_wizard, "step_profile_required", lambda *a, **k: {})
    monkeypatch.setattr(setup_wizard, "ask_yes_no", lambda *a, **k: False)
    monkeypatch.setattr(setup_wizard, "load_existing_profile", lambda: {})
    choices = iter((2, 6))
    monkeypatch.setattr(setup_wizard, "ask_choice", lambda *a, **k: next(choices))

    with pytest.raises(FilesystemProtectionError, match="unsafe path component"):
        setup_wizard.wizard_edit({}, {})

    assert target.read_text(encoding="utf-8") == original
    assert _restriction_snapshot(target) == before


def test_c21_log_handler_is_not_created_for_symlinked_parent(
    tmp_path: Path, isolated_logging: list[logging.Handler], monkeypatch
) -> None:
    outside, link = _symlinked_parent(tmp_path)
    target = outside / "agent.log"
    original = "previous diagnostic\n"
    target.write_text(original, encoding="utf-8")
    before = _restriction_snapshot(target)

    def forbidden_handler(*args, **kwargs):
        raise AssertionError("RotatingFileHandler created before path validation")

    monkeypatch.setattr(logging_setup, "RotatingFileHandler", forbidden_handler)

    with pytest.raises(FilesystemProtectionError, match="unsafe path component"):
        configure_logging(link / "agent.log")

    assert target.read_text(encoding="utf-8") == original
    assert _restriction_snapshot(target) == before


@pytest.mark.parametrize("backend_type", [CloakBrowserBackend, PlaywrightBrowserBackend])
def test_c21_browser_profile_is_not_created_through_symlinked_parent(
    tmp_path: Path, backend_type: type
) -> None:
    outside, link = _symlinked_parent(tmp_path)
    profile = outside / "profile"
    profile.mkdir()
    marker = profile / "Preferences"
    original = '{"persisted": true}'
    marker.write_text(original, encoding="utf-8")
    before = _restriction_snapshot(marker)
    launched: list[Path] = []

    async def launcher(profile_dir, *, headless: bool):
        launched.append(Path(profile_dir))
        return object()

    backend = backend_type(link / "profile", False, launcher=launcher)

    with pytest.raises(Exception) as raised:
        asyncio.run(backend.start())

    assert "filesystem_protection_failed" in str(raised.value)
    assert "error_type=FilesystemProtectionError" in str(raised.value)
    assert launched == []
    assert marker.read_text(encoding="utf-8") == original
    assert _restriction_snapshot(marker) == before


def test_c21_env_is_not_read_through_symlinked_parent(
    tmp_path: Path, monkeypatch
) -> None:
    outside, link = _symlinked_parent(tmp_path)
    env_file = outside / ".env"
    original = "\n".join(f"{key}={value}" for key, value in VALID_ENV.items()) + "\n"
    env_file.write_text(original, encoding="utf-8")
    before = _restriction_snapshot(env_file)

    def forbidden_read(*args, **kwargs):
        raise AssertionError(".env was read before path validation")

    monkeypatch.setattr(config, "dotenv_values", forbidden_read)

    with pytest.raises(FilesystemProtectionError, match="unsafe path component"):
        load_settings(env_path=link / ".env", environ=None)

    assert env_file.read_text(encoding="utf-8") == original
    assert _restriction_snapshot(env_file) == before


def test_c21_profile_is_not_read_through_symlinked_parent(
    tmp_path: Path, monkeypatch
) -> None:
    outside, link = _symlinked_parent(tmp_path)
    profile_file = write_profile(outside)
    original = profile_file.read_text(encoding="utf-8")
    before = _restriction_snapshot(profile_file)

    def forbidden_load(*args, **kwargs):
        raise AssertionError("profile.yaml was read before path validation")

    monkeypatch.setattr(config.yaml, "safe_load", forbidden_load)

    with pytest.raises(FilesystemProtectionError, match="unsafe path component"):
        load_settings(
            profile_path=link / "profile.yaml",
            environ=VALID_ENV,
        )

    assert profile_file.read_text(encoding="utf-8") == original
    assert _restriction_snapshot(profile_file) == before


def test_c21_main_profile_root_is_not_created_through_symlinked_parent(
    tmp_path: Path,
) -> None:
    outside, link = _symlinked_parent(tmp_path)
    profile_root = outside / "profiles"
    profile_root.mkdir()
    marker = profile_root / "hh-account-1"
    marker.mkdir()
    before = _restriction_snapshot(marker)

    class _Settings:
        database_path = tmp_path / "state.db"
        browser_profile_dir = link / "profiles"

    with pytest.raises(FilesystemProtectionError, match="unsafe path component"):
        asyncio.run(main.run(_Settings(), telegram_user_id=42))

    assert marker.is_dir()
    assert _restriction_snapshot(marker) == before


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_c21_windows_junction_is_rejected_by_preflight(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "agent.db"
    original = "session-data"
    marker.write_text(original, encoding="utf-8")
    before = _restriction_snapshot(marker)
    junction = tmp_path / "junction"

    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"junction creation unavailable: {result.stderr.strip()[:120]}")

    try:
        with pytest.raises(FilesystemProtectionError, match="unsafe path component"):
            security_utils.validate_path_before_io(
                junction / "agent.db", kind="test path"
            )
        assert marker.read_text(encoding="utf-8") == original
        assert _restriction_snapshot(marker) == before
    finally:
        if junction.exists():
            junction.rmdir()


# ---------------------------------------------------------------------------
# C22 — Full-audit regressions for placeholder, URI and identity edge cases
# ---------------------------------------------------------------------------


def test_c22_json_authorization_headers_cannot_hide_tails() -> None:
    cases = (
        '{"Authorization": "[REDACTED] ACTUAL-JSON-AUTH"}',
        '{"Proxy-Authorization": "Basic [REDACTED]ACTUAL-JSON-PROXY"}',
        "client_id=[REDACTED]ACTUAL-STANDALONE-CLIENT",
    )

    for text in cases:
        redacted = redact_sensitive_text(text)
        assert "ACTUAL-" not in redacted
        assert "[REDACTED]" in redacted
        assert redact_sensitive_text(redacted) == redacted


def test_c15_relative_sqlite_uri_cannot_follow_later_cwd_change(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    uri = "file:relative-state.db?mode=rwc"
    target = security_utils.resolve_sqlite_target(uri)
    assert target.uri is True
    assert target.path is not None
    assert target.path.parent == tmp_path
    assert target.connect_name.startswith("file:///")

    database = Database(uri)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)

    database.init()

    assert (tmp_path / "relative-state.db").is_file()
    assert not (other / "relative-state.db").exists()


def test_c15_database_connect_protects_existing_sqlite_side_files(
    tmp_path: Path, monkeypatch
) -> None:
    db_path = tmp_path / "agent.db"
    side_file = tmp_path / "agent.db-sqlite"
    side_file.write_text("sqlite-side-file", encoding="utf-8")
    calls: list[tuple[Path, bool]] = []
    original = security_utils.protect_sqlite_siblings

    def tracking(path):
        candidate = Path(path)
        calls.append((candidate, candidate.exists()))
        return original(candidate)

    monkeypatch.setattr(database_module, "protect_sqlite_siblings", tracking)

    Database(db_path).init()

    assert any(path == db_path and existed for path, existed in calls)
    assert side_file.read_text(encoding="utf-8") == "sqlite-side-file"
    _assert_restricted(side_file)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_c21_dotdot_through_junction_is_rejected_before_normalization(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    junction = tmp_path / "junction"
    safe = tmp_path / "safe"
    safe.mkdir()
    safe_file = safe / "agent.db"
    original = "do-not-touch"
    safe_file.write_text(original, encoding="utf-8")
    before = _restriction_snapshot(safe_file)

    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"junction creation unavailable: {result.stderr.strip()[:120]}")

    try:
        dangerous = junction / ".." / "safe" / "agent.db"
        with pytest.raises(FilesystemProtectionError, match="unsafe path component"):
            security_utils.validate_path_before_io(dangerous, kind="test path")
        assert safe_file.read_text(encoding="utf-8") == original
        assert _restriction_snapshot(safe_file) == before
    finally:
        if junction.exists():
            junction.rmdir()


class _FakeApprovalService:
    def __init__(self, sender: object) -> None:
        self.sender = sender

    async def approve_and_apply(self, job_id: str, telegram_user_id: int):
        raise AssertionError("approval callback reached another user's session")

    def skip(self, job_id: str, telegram_user_id: int):
        raise AssertionError("skip callback reached another user's session")


class _FakeCallback:
    def __init__(self, user_id: int, data: str) -> None:
        self.from_user = SimpleNamespace(id=user_id)
        self.data = data
        self.message = None
        self.answers: list[tuple[str, bool]] = []

    async def answer(self, text: str, *, show_alert: bool = False) -> None:
        self.answers.append((text, show_alert))


def test_c22_callback_from_foreign_user_is_rejected_before_approval(
    tmp_path: Path,
) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    database.create_user(42)
    database.create_user(43)
    client = SimpleNamespace(telegram_user_id=42)
    context = SimpleNamespace(telegram_user_id=42)
    service = tg_bot.TelegramService(
        settings,
        database,
        _FakeApprovalService(client),
        contexts={42: context},
        bot=SimpleNamespace(),
        hh_client=client,
    )
    callback = _FakeCallback(43, "apply:job-1")

    asyncio.run(service._callback_handler(callback))

    assert callback.answers == [(tg_bot.NO_RUNTIME_REPLY, True)]


def test_c22_callback_with_foreign_approval_sender_is_rejected(
    tmp_path: Path,
) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    database.create_user(42)
    client = SimpleNamespace(telegram_user_id=42)
    foreign_sender = SimpleNamespace(telegram_user_id=43)
    context = SimpleNamespace(telegram_user_id=42)
    service = tg_bot.TelegramService(
        settings,
        database,
        _FakeApprovalService(foreign_sender),
        contexts={42: context},
        bot=SimpleNamespace(),
        hh_client=client,
    )
    callback = _FakeCallback(42, "apply:job-1")

    asyncio.run(service._callback_handler(callback))

    assert callback.answers == [(tg_bot.NO_RUNTIME_REPLY, True)]


def test_c22_submit_rejects_permit_for_another_user() -> None:
    client = object.__new__(hh_client_module.HHClient)
    client.telegram_user_id = 42
    client.session_lock = None
    permission = ApplicationPermission("job-1", "permit", 43)

    assert asyncio.run(client.submit_application(permission)) is False


def test_c22_skip_rejects_foreign_pending_vacancy(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    database.init()
    database.create_user(42)
    database.create_user(43)
    with database._connect() as connection:
        connection.execute(
            """
            INSERT INTO vacancies
                (id, telegram_user_id, title, url, status, discovered_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "job-1",
                42,
                "vacancy",
                "https://hh.kz/vacancy/job-1",
                VacancyStatus.PENDING_APPROVAL.value,
                "2026-01-01T00:00:00Z",
            ),
        )

    assert database.skip("job-1", 43, 43) is False
    assert (
        database.get("job-1", telegram_user_id=42).status
        == VacancyStatus.PENDING_APPROVAL
    )

    assert database.skip("job-1", 42, 42) is True
    assert (
        database.get("job-1", telegram_user_id=42).status == VacancyStatus.SKIPPED
    )


# ---------------------------------------------------------------------------
# DEF-1 — TOCTOU: validation must be tied to the object that is opened
# ---------------------------------------------------------------------------


def test_def1_write_is_not_redirected_by_a_late_parent_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "agent.env"
    outside_file.write_text("outside-secret", encoding="utf-8")
    before = _restriction_snapshot(outside_file)
    parent = tmp_path / "config"
    parent.mkdir()
    target = parent / ".env"

    original_validate = security_utils.validate_path_before_io
    calls = {"count": 0}

    def swap_after_first_validation(path, *, kind="path"):
        result = original_validate(path, kind=kind)
        calls["count"] += 1
        if calls["count"] == 1:
            _replace_with_link(parent, outside)
        return result

    monkeypatch.setattr(
        security_utils, "validate_path_before_io", swap_after_first_validation
    )

    with pytest.raises(FilesystemProtectionError):
        security_utils.write_validated_text(
            target, f"token={SECRET}\n", kind="configuration file"
        )

    assert outside_file.read_text(encoding="utf-8") == "outside-secret"
    assert _restriction_snapshot(outside_file) == before
    assert not (outside / ".env").exists()


def test_def1_read_is_not_redirected_by_a_late_parent_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "profile.yaml"
    outside_file.write_text(f"token: {SECRET}\n", encoding="utf-8")
    parent = tmp_path / "config"
    parent.mkdir()
    target = parent / "profile.yaml"

    original_validate = security_utils.validate_path_before_io
    calls = {"count": 0}

    def swap_after_first_validation(path, *, kind="path"):
        result = original_validate(path, kind=kind)
        calls["count"] += 1
        if calls["count"] == 1:
            _replace_with_link(parent, outside)
        return result

    monkeypatch.setattr(
        security_utils, "validate_path_before_io", swap_after_first_validation
    )

    with pytest.raises(FilesystemProtectionError):
        security_utils.read_validated_text(target, kind="configuration file")


def test_def1_log_handler_open_is_not_redirected_by_a_late_parent_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "agent.log"
    outside_file.write_text("unrelated-log", encoding="utf-8")
    before = _restriction_snapshot(outside_file)
    parent = tmp_path / "logs"
    parent.mkdir()

    original_validate = security_utils.validate_path_before_io
    calls = {"count": 0}

    def swap_after_first_validation(path, *, kind="path"):
        result = original_validate(path, kind=kind)
        calls["count"] += 1
        if calls["count"] == 1:
            _replace_with_link(parent, outside)
        return result

    monkeypatch.setattr(
        security_utils, "validate_path_before_io", swap_after_first_validation
    )

    with pytest.raises(FilesystemProtectionError):
        logging_setup.RotatingFileHandler(parent / "agent.log")

    assert outside_file.read_text(encoding="utf-8") == "unrelated-log"
    assert _restriction_snapshot(outside_file) == before


def test_def1_database_guard_rejects_a_late_database_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "agent.db"
    outside_file.write_text("outside-session-data", encoding="utf-8")
    before = _restriction_snapshot(outside_file)
    parent = tmp_path / "state"
    parent.mkdir()
    target = parent / "agent.db"

    original_validate = security_utils.validate_path_before_io
    calls = {"count": 0}

    def swap_after_first_validation(path, *, kind="path"):
        result = original_validate(path, kind=kind)
        calls["count"] += 1
        if calls["count"] == 1:
            _replace_with_link(parent, outside)
        return result

    monkeypatch.setattr(
        security_utils, "validate_path_before_io", swap_after_first_validation
    )

    with pytest.raises(FilesystemProtectionError):
        database_module.Database(target).init()

    assert outside_file.read_text(encoding="utf-8") == "outside-session-data"
    assert _restriction_snapshot(outside_file) == before


# ---------------------------------------------------------------------------
# DEF-2 — SQLite sibling scanning must not follow symlinks/reparse points
# ---------------------------------------------------------------------------


def test_def2_regular_sqlite_side_file_is_protected_without_following_links(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "state.db"
    db_path.write_text("database", encoding="utf-8")
    side = db_path.with_name("state.db-wal")
    side.write_text("journal", encoding="utf-8")

    security_utils.protect_sqlite_siblings(db_path)

    _assert_restricted(db_path)
    _assert_restricted(side)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink test")
def test_def2_posix_symlink_sqlite_side_file_is_rejected_before_protection(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "unrelated.db"
    outside_file.write_text("unrelated-session-data", encoding="utf-8")
    before = _restriction_snapshot(outside_file)

    db_path = tmp_path / "state.db"
    db_path.write_text("database", encoding="utf-8")
    side = db_path.with_name("state.db-journal")
    _make_symlink(side, outside_file)

    with pytest.raises(FilesystemProtectionError, match="unsafe SQLite side file"):
        security_utils.protect_sqlite_siblings(db_path)

    assert side.is_symlink()
    assert outside_file.read_text(encoding="utf-8") == "unrelated-session-data"
    assert _restriction_snapshot(outside_file) == before


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_def2_windows_junction_sqlite_side_file_is_rejected_before_protection(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "unrelated.db"
    marker.write_text("unrelated-session-data", encoding="utf-8")

    db_path = tmp_path / "state.db"
    db_path.write_text("database", encoding="utf-8")
    side = db_path.with_name("state.db-journal")
    _replace_with_link(side, outside)
    try:
        with pytest.raises(
            FilesystemProtectionError, match="unsafe SQLite side file"
        ):
            security_utils.protect_sqlite_siblings(db_path)

        assert marker.read_text(encoding="utf-8") == "unrelated-session-data"
    finally:
        if side.exists():
            side.rmdir()


# ---------------------------------------------------------------------------
# DEF-3 — RedactingFilter must not mutate a shared LogRecord
# ---------------------------------------------------------------------------


def test_def3_two_handlers_and_sequential_records_do_not_mutate_original() -> None:
    original = logging.LogRecord(
        "phase57",
        logging.WARNING,
        __file__,
        1,
        "token=%s cookie=%s",
        (SECRET, SECRET),
        (RuntimeError, RuntimeError(f"Authorization: Bearer {SECRET}"), None),
        stack_info=f"stack {SECRET}",
    )
    saved = (
        original.msg,
        original.args,
        original.exc_info,
        original.stack_info,
    )

    streams = [StringIO(), StringIO()]
    for stream in streams:
        handler = logging.StreamHandler(stream)
        handler.addFilter(RedactingFilter())
        handler.setFormatter(logging.Formatter("%(message)s"))
        for record in (original, original):
            handler.handle(record)

    for stream in streams:
        logged = stream.getvalue()
        assert SECRET not in logged
        assert logged.count("[REDACTED]") >= 2
        assert "exception=RuntimeError" in logged
        assert "Traceback" not in logged

    assert (original.msg, original.args, original.exc_info, original.stack_info) == saved
    assert original.exc_info is not None


def test_def3_log_record_replacement_compat_matches_supported_python() -> None:
    compat_installed = bool(
        getattr(logging.Logger.handle, "_phase57_record_replacement", False)
    )
    if sys.version_info < (3, 12):
        assert compat_installed
        assert getattr(
            logging.Handler.handle, "_phase57_record_replacement", False
        )
    else:
        assert not compat_installed


def test_def3_python311_handle_shim_replaces_record_from_filter() -> None:
    original = logging.LogRecord(
        "phase57.compat",
        logging.WARNING,
        __file__,
        1,
        "token=%s",
        (SECRET,),
        (RuntimeError, RuntimeError(f"Bearer {SECRET}"), None),
        stack_info=f"stack {SECRET}",
    )
    saved = (
        original.msg,
        original.args,
        original.exc_info,
        original.stack_info,
    )
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter())
    handler.setFormatter(logging.Formatter("%(message)s"))

    logging_setup._handler_handle_with_record_replacement(handler, original)

    logged = stream.getvalue()
    assert SECRET not in logged
    assert "[REDACTED]" in logged
    assert "exception=RuntimeError" in logged
    assert "Traceback" not in logged
    assert (original.msg, original.args, original.exc_info, original.stack_info) == saved


def test_def3_logger_handle_shim_forwards_redacted_record_without_mutation() -> None:
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter())
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("phase57.compat.logger")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    original = logging.LogRecord(
        logger.name,
        logging.WARNING,
        __file__,
        1,
        "token=%s",
        (SECRET,),
        None,
    )
    saved = (original.msg, original.args, original.exc_info, original.stack_info)

    logger.handle(original)

    logged = stream.getvalue()
    assert SECRET not in logged
    assert "[REDACTED]" in logged
    assert (original.msg, original.args, original.exc_info, original.stack_info) == saved


# ---------------------------------------------------------------------------
# DEF-4 — client_id is always redacted, independent of a context window
# ---------------------------------------------------------------------------


def test_def4_client_id_is_redacted_in_json_multiline_and_reordered_fields() -> None:
    text = "\n".join(
        (
            '{"status":"ok","client_id":"JSON-CLIENT","request_id":"req-1"}',
            "client_id=MULTILINE-CLIENT",
            "request_id=req-2 status=ok client_id=REORDERED-CLIENT",
        )
    )

    redacted = redact_sensitive_text(text)

    for value in ("JSON-CLIENT", "MULTILINE-CLIENT", "REORDERED-CLIENT"):
        assert value not in redacted
    assert redacted.count("[REDACTED]") == 3
    assert "req-1" in redacted
    assert "req-2" in redacted
    assert "status=ok" in redacted
