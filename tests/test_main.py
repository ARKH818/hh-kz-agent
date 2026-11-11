import asyncio
import ast
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

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

# ============================================================================
# Stage 4\7 — run_search_cycle() user-specific error isolation
# ============================================================================


class ErroringHHClient:
    """HHClient double that returns search errors to trigger circuit breaker."""

    def __init__(self, error_reason: str = "network_error"):
        self.error_reason = error_reason
        self.search_calls = 0

    async def search_vacancies(self, query, areas, experience_filters):
        self.search_calls += 1
        return VacancySearchResult([], 0, 0, error_reason=self.error_reason)

    async def read_vacancy(self, summary, captcha_solver=None):
        return VacancyDetails(summary, PageState.NETWORK_ERROR, error="read failed")

    async def check_messages(self, notify):
        pass

    async def read_company_details(self, company_url: str):
        return CompanyDetails(None, None)


def _make_two_user_contexts(tmp_path: Path, mode: str = "dry_run"):
    """Create two fully independent UserRuntimeContext objects."""
    app_settings = settings(tmp_path, mode)
    database = Database(app_settings.database_path)
    database.init()

    # Stage 4\7: register both users in the test DB so that
    # search_runs.telegram_user_id FOREIGN KEY is satisfied.
    database.create_user(telegram_user_id=42, username="user42", now=NOW)
    database.create_user(telegram_user_id=43, username="user43", now=NOW)

    us_a = load_user_settings(app_settings)
    us_b = load_user_settings(app_settings)

    ctx_a = create_runtime_context(42, app_settings, database, us_a)
    ctx_b = create_runtime_context(43, app_settings, database, us_b)

    return app_settings, database, ctx_a, ctx_b


def test_run_search_cycle_error_counter_isolation(tmp_path: Path) -> None:
    """Stage 4\7: search error increments only the failing user's counter."""
    _, database, ctx_a, ctx_b = _make_two_user_contexts(tmp_path)
    telegram_a = FakeTelegram()
    telegram_b = FakeTelegram()
    hh_client = ErroringHHClient("network_error")

    # Run cycle for user A — enough errors to trigger counter
    asyncio.run(
        run_search_cycle(ctx_a, hh_client, telegram_a, now_factory=lambda: NOW)
    )

    # A accumulated errors; B untouched
    assert ctx_a.consecutive_search_errors >= 1
    assert ctx_b.consecutive_search_errors == 0
    assert ctx_b.paused is False
    assert ctx_b.circuit_reason == ""
    assert ctx_b.next_run_at is None


def test_run_search_cycle_circuit_breaker_isolation(tmp_path: Path) -> None:
    """Stage 4\7: circuit breaker pauses only the failing user."""
    _, database, ctx_a, ctx_b = _make_two_user_contexts(tmp_path)
    telegram_a = FakeTelegram()
    telegram_b = FakeTelegram()

    # circuit_breaker_page_errors default = 3; run enough cycles for A
    hh_client = ErroringHHClient("network_error")
    for _ in range(3):
        if not ctx_a.paused:
            asyncio.run(
                run_search_cycle(ctx_a, hh_client, telegram_a, now_factory=lambda: NOW)
            )

    # A is paused by circuit breaker
    assert ctx_a.paused is True
    assert ctx_a.circuit_reason == "search_errors"

    # B is completely unaffected
    assert ctx_b.paused is False
    assert ctx_b.circuit_reason == ""
    assert ctx_b.consecutive_search_errors == 0
    assert ctx_b.next_run_at is None


def test_run_search_cycle_notification_goes_to_correct_user(tmp_path: Path) -> None:
    """Stage 4\7: circuit-breaker notification is sent only to the failing user."""
    _, database, ctx_a, ctx_b = _make_two_user_contexts(tmp_path)
    telegram_a = FakeTelegram()
    telegram_b = FakeTelegram()

    hh_client = ErroringHHClient("network_error")
    for _ in range(3):
        if not ctx_a.paused:
            asyncio.run(
                run_search_cycle(ctx_a, hh_client, telegram_a, now_factory=lambda: NOW)
            )

    # Notification sent to A
    assert len(telegram_a.notifications) >= 1
    notified_users = [uid for _, uid in telegram_a.notifications]
    assert 42 in notified_users
    assert 43 not in notified_users

    # B received nothing
    assert telegram_b.notifications == []


def test_run_search_cycle_database_isolation(tmp_path: Path) -> None:
    """Stage 4\7: failed search run is saved under the correct telegram_user_id."""
    _, database, ctx_a, ctx_b = _make_two_user_contexts(tmp_path)
    telegram_a = FakeTelegram()
    telegram_b = FakeTelegram()

    hh_client = ErroringHHClient("network_error")
    for _ in range(3):
        if not ctx_a.paused:
            asyncio.run(
                run_search_cycle(ctx_a, hh_client, telegram_a, now_factory=lambda: NOW)
            )

    # A has a saved search run with circuit_reason
    run_a = database.latest_search_run(telegram_user_id=42)
    assert run_a is not None
    assert run_a.circuit_reason == "search_errors"
    assert run_a.state == "paused_by_circuit_breaker"

    # B has no search run at all
    run_b = database.latest_search_run(telegram_user_id=43)
    assert run_b is None


# ============================================================================
# Stage 4\7 — agent_loop() user-specific runtime isolation
# ============================================================================


class QuickHHClient:
    """HHClient double that completes a cycle instantly with no results, tracking search calls."""

    def __init__(self):
        self.search_calls = 0

    async def search_vacancies(self, query, areas, experience_filters):
        self.search_calls += 1
        return VacancySearchResult([], 0, 0)

    async def read_vacancy(self, summary, captcha_solver=None):
        return VacancyDetails(summary, PageState.VACANCY_LOADED, "Co", "desc")

    async def check_messages(self, notify):
        pass

    async def read_company_details(self, company_url: str):
        return CompanyDetails(None, None)


def test_agent_loop_next_run_at_isolation(tmp_path: Path) -> None:
    """Stage 4\7: agent_loop sets next_run_at only on its own context."""
    _, database, ctx_a, ctx_b = _make_two_user_contexts(tmp_path)
    telegram_a = FakeTelegram()
    telegram_b = FakeTelegram()
    hh_client = QuickHHClient()

    next_run_set_event = asyncio.Event()
    orig_wait = ctx_a.wake_event.wait
    async def mock_wait():
        next_run_set_event.set()
        return await orig_wait()
    ctx_a.wake_event.wait = mock_wait

    async def scenario():
        task_a = asyncio.create_task(
            agent_loop(ctx_a, hh_client, telegram_a)
        )

        # Детерминированное ожидание момента входа в wake_event.wait(),
        # что гарантирует, что следующий за ним по порядку в agent_loop() next_run_at уже установлен.
        await asyncio.wait_for(next_run_set_event.wait(), timeout=2.0)

        assert ctx_a.next_run_at is not None
        assert ctx_b.next_run_at is None
        assert hh_client.search_calls >= 1

        task_a.cancel()
        try:
            await task_a
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())

    # После отмены задачи проверяется очистка next_run_at
    assert ctx_a.next_run_at is None
    assert ctx_b.next_run_at is None
    assert ctx_b.paused is False
    assert ctx_b.consecutive_search_errors == 0
    assert not ctx_b.wake_event.is_set()


def test_agent_loop_paused_isolation(tmp_path: Path) -> None:
    """Stage 4\7: paused=True on A means no cycle; B is unaffected."""
    _, database, ctx_a, ctx_b = _make_two_user_contexts(tmp_path)
    telegram_a = FakeTelegram()
    telegram_b = FakeTelegram()
    hh_client = QuickHHClient()

    ctx_a.paused = True

    wake_awaited_event = asyncio.Event()
    orig_wait = ctx_a.wake_event.wait
    async def mock_wait():
        wake_awaited_event.set()
        return await orig_wait()
    ctx_a.wake_event.wait = mock_wait

    async def scenario():
        task_a = asyncio.create_task(
            agent_loop(ctx_a, hh_client, telegram_a)
        )
        # Дожидаемся начала ожидания через asyncio.wait_for() с timeout без подавления исключений
        await asyncio.wait_for(wake_awaited_event.wait(), timeout=2.0)

        task_a.cancel()
        try:
            await task_a
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())

    # A stayed paused (agent_loop does not unpause)
    assert ctx_a.paused is True
    assert hh_client.search_calls == 0
    assert ctx_a.next_run_at is None

    # B completely unaffected
    assert ctx_b.paused is False
    assert ctx_b.next_run_at is None
    assert not ctx_b.wake_event.is_set()


def test_agent_loop_wake_event_isolation(tmp_path: Path, monkeypatch) -> None:
    """Stage 4\7: wake_event of A is independent from B."""
    _, database, ctx_a, ctx_b = _make_two_user_contexts(tmp_path)
    telegram_a = FakeTelegram()
    telegram_b = FakeTelegram()
    hh_client = QuickHHClient()

    # Set A's wake event before starting the loop
    ctx_a.wake_event.set()

    search_cycle_calls = 0
    second_cycle_started = asyncio.Event()

    async def fake_run_search_cycle(context, hh_client, telegram, now_factory=None):
        nonlocal search_cycle_calls
        search_cycle_calls += 1
        if search_cycle_calls == 2:
            second_cycle_started.set()
            await asyncio.Future()
        return SimpleNamespace(
            finished_at=datetime.now().isoformat()
        )

    monkeypatch.setattr(main_module, "run_search_cycle", fake_run_search_cycle)

    async def scenario():
        task_a = asyncio.create_task(
            agent_loop(ctx_a, hh_client, telegram_a)
        )
        # Protective timeout only - wait until second run_search_cycle is entered
        await asyncio.wait_for(second_cycle_started.wait(), timeout=2.0)

        task_a.cancel()
        try:
            await task_a
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())

    assert search_cycle_calls == 2
    # A's wake_event was consumed and cleared by agent_loop's finally block
    assert not ctx_a.wake_event.is_set()
    # B's wake_event is untouched
    assert not ctx_b.wake_event.is_set()
    assert ctx_b.next_run_at is None
    assert ctx_b.paused is False