import asyncio
import ast
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

import main as main_module
from config import load_settings
from database import Database, VacancyStatus
from hh_client import (
    CompanyDetails,
    PageState,
    VacancyDetails,
    VacancySearchResult,
    VacancySummary,
)
from main import (
    SearchRunStats,
    VacancyProcessResult,
    agent_loop,
    process_vacancy,
    run_search_cycle,
)
from config import load_user_settings
from runtime_context import create_runtime_context
from tests.test_config import VALID_ENV, write_profile
from tg_bot import TelegramService

NOW = datetime(2026, 7, 26, 9, 0, tzinfo=UTC)
SUMMARY = VacancySummary(
    "job-1", "Python developer", "https://example.com/vacancy/job-1", "Python"
)


class FakeHHClient:
    def __init__(
        self,
        details: VacancyDetails,
        search_results: list[VacancySearchResult] | None = None,
    ):
        self.details = details
        self.search_results = search_results or []
        self.read_calls = 0
        self.search_calls = 0
        self.message_checks = 0

    async def read_vacancy(self, summary, captcha_solver=None):
        self.read_calls += 1
        return self.details

    async def search_vacancies(self, query, areas, experience_filters):
        self.search_calls += 1
        return (
            self.search_results.pop(0)
            if self.search_results
            else VacancySearchResult([], 0, 0)
        )

    async def check_messages(self, notify):
        self.message_checks += 1

    async def read_company_details(self, company_url: str):
        return CompanyDetails(4.7, 128)


class SearchReached(Exception):
    pass


class StopAtSearchHHClient(FakeHHClient):
    async def search_vacancies(self, *_args):
        raise SearchReached


class FakeTelegram:
    def __init__(self):
        self.previews: list[tuple[str, bool, int | None]] = []
        self.notifications: list[tuple[str, int | None]] = []

    async def send_preview(self, vacancy, include_actions: bool, telegram_user_id: int | None = None):
        self.previews.append((vacancy.id, include_actions, telegram_user_id))

    async def notify(self, text: str, telegram_user_id: int | None = None):
        self.notifications.append((text, telegram_user_id))

    async def request_captcha(self, screenshot, title: str, timeout_seconds: int, telegram_user_id: int | None = None):
        return None


def settings(tmp_path: Path, mode: str):
    loaded = load_settings(
        profile_path=write_profile(tmp_path),
        environ={**VALID_ENV, "APP_MODE": mode},
    )
    return replace(loaded, database_path=tmp_path / "agent.db")


def test_dry_run_records_and_previews_to_previewed_status(tmp_path: Path) -> None:
    app_settings = settings(tmp_path, "dry_run")
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    context = create_runtime_context(42, app_settings, database)

    asyncio.run(
        process_vacancy(
            context,
            FakeHHClient(details),
            telegram,
            SUMMARY,
            now_factory=lambda: NOW,
        )
    )

    vacancy = database.get("job-1", telegram_user_id=42)
    assert vacancy.status is VacancyStatus.PREVIEWED
    assert vacancy.company_rating == 4.7
    assert telegram.previews == [("job-1", False, 42)]


def test_approval_mode_records_pending_and_sends_actions(tmp_path: Path) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    context = create_runtime_context(42, app_settings, database)

    asyncio.run(
        process_vacancy(
            context,
            FakeHHClient(details),
            telegram,
            SUMMARY,
            now_factory=lambda: NOW,
        )
    )

    assert database.get("job-1", telegram_user_id=42).status is VacancyStatus.PENDING_APPROVAL
    assert telegram.previews == [("job-1", True, 42)]


def test_browser_read_error_is_persisted_as_apply_failed(tmp_path: Path) -> None:
    app_settings = settings(tmp_path, "dry_run")
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.NETWORK_ERROR, error="navigation failed"
    )
    context = create_runtime_context(42, app_settings, database)

    asyncio.run(
        process_vacancy(
            context,
            FakeHHClient(details),
            telegram,
            SUMMARY,
            now_factory=lambda: NOW,
        )
    )

    vacancy = database.get("job-1", telegram_user_id=42)
    assert vacancy.status is VacancyStatus.APPLY_FAILED
    assert "navigation failed" in vacancy.error_text


def test_cli_reports_configuration_error_without_traceback(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--check-config",
            "--env-file",
            str(tmp_path / "missing.env"),
            "--profile",
            str(tmp_path / "missing.yaml"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stderr.startswith("Configuration error:")
    assert "Traceback" not in result.stderr


# ============================================================================
# Stage 4\3 focused tests — no global TG_USER_ID
# ============================================================================

def test_run_does_not_use_global_user_id(tmp_path: Path) -> None:
    """run(settings) does not depend on settings.tg_user_id (removed)."""
    app_settings = settings(tmp_path, "dry_run")
    # settings no longer has tg_user_id attribute
    assert not hasattr(app_settings, "tg_user_id")


def test_run_with_telegram_user_id_creates_context_with_that_id(tmp_path: Path) -> None:
    """run(settings, telegram_user_id=42) would create runtime context with ID 42."""
    app_settings = settings(tmp_path, "dry_run")
    database = Database(app_settings.database_path)
    database.init()

    context = create_runtime_context(telegram_user_id=42, settings=app_settings, database=database)

    assert context.telegram_user_id == 42
    assert context.settings is app_settings
    assert context.database is database


def test_cli_accepts_telegram_user_id_argument() -> None:
    """CLI parser accepts --telegram-user-id."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--telegram-user-id", type=int, default=None)
    args = parser.parse_args(["--telegram-user-id", "42"])
    assert args.telegram_user_id == 42


def test_cli_without_telegram_user_id_has_none_default() -> None:
    """CLI without --telegram-user-id defaults to None."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--telegram-user-id", type=int, default=None)
    args = parser.parse_args([])
    assert args.telegram_user_id is None


# ============================================================================
# Stage 4\5: production HHClient receives telegram_user_id
# ============================================================================

def test_main_run_passes_telegram_user_id_to_hh_client() -> None:
    """Production main.run() должен передавать telegram_user_id в HHClient.

    Проверка через AST, что HHClient вызывается с telegram_user_id.
    """
    source = Path("main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    hh_client_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "HHClient":
                hh_client_calls.append(node)

    assert len(hh_client_calls) >= 1, "main.py must call HHClient"

    # Проверяем, что хотя бы один вызов передаёт telegram_user_id
    found_telegram_user_id = False
    for call in hh_client_calls:
        # Позиционный аргумент (5-й аргумент, индекс 4)
        if len(call.args) >= 5:
            found_telegram_user_id = True
        # Или keyword argument
        for keyword in call.keywords:
            if keyword.arg == "telegram_user_id":
                found_telegram_user_id = True

    assert found_telegram_user_id, (
        "HHClient must be called with telegram_user_id in main.run()"
    )

# ============================================================================
# Stage 4\6 — one UserSettings instance per user across components
# ============================================================================

def test_run_none_telegram_user_id_polling_only(tmp_path: Path, monkeypatch) -> None:
    """Stage 4\3: run(settings, telegram_user_id=None) starts polling only.

    No agent loop, no user-specific runtime components, no exception
    solely because the user ID is missing.
    """
    import main as m

    app_settings = settings(tmp_path, "dry_run")

    started_polling: list[bool] = []
    agent_loops: list[int] = []

    class FakeBackend:
        async def start(self):
            return object()

        async def close(self):
            return None

    class FakeTelegram:
        def __init__(self, *args, **kwargs):
            self.contexts = kwargs.get("contexts")

        async def start_polling(self):
            started_polling.append(True)

        async def stop(self):
            return None

    async def fake_agent_loop(context, hh_client, telegram):
        agent_loops.append(context.telegram_user_id)

    monkeypatch.setattr(m, "create_browser_backend", lambda s: FakeBackend())
    monkeypatch.setattr(m, "TelegramService", FakeTelegram)
    monkeypatch.setattr(m, "agent_loop", fake_agent_loop)

    asyncio.run(m.run(app_settings, telegram_user_id=None))

    assert started_polling == [True]
    assert agent_loops == []


def test_single_user_settings_shared_across_components(tmp_path: Path) -> None:
    """HHClient, ApprovalGuard, ApprovalService and context share ONE instance."""
    from approval import ApprovalGuard, ApprovalService
    from hh_client import HHClient

    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()

    user_settings = load_user_settings(app_settings)
    context = create_runtime_context(
        42, app_settings, database, user_settings
    )

    guard = ApprovalGuard(context.user_settings, database)
    hh_client = HHClient(
        object(), app_settings, database, guard,
        telegram_user_id=42, user_settings=context.user_settings,
    )
    approval_service = ApprovalService(
        context.user_settings, database, hh_client
    )

    assert context.user_settings is user_settings
    assert hh_client.user_settings is context.user_settings
    assert guard.user_settings is context.user_settings
    assert approval_service.user_settings is context.user_settings


def test_process_vacancy_uses_context_user_settings(tmp_path: Path) -> None:
    """process_vacancy reads app_mode from context.user_settings."""
    from dataclasses import replace as dc_replace

    app_settings = settings(tmp_path, "dry_run")
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    user_settings = load_user_settings(app_settings)
    context = create_runtime_context(
        42, app_settings, database, user_settings
    )
    # Per-user override via immutable replace + rebind
    context.user_settings = dc_replace(context.user_settings, app_mode="approval")

    asyncio.run(
        process_vacancy(
            context,
            FakeHHClient(details),
            telegram,
            SUMMARY,
            now_factory=lambda: NOW,
        )
    )

    assert database.get("job-1", telegram_user_id=42).status is VacancyStatus.PENDING_APPROVAL
    assert telegram.previews == [("job-1", True, 42)]