import asyncio
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
from tests.test_config import VALID_ENV, write_profile
from tg_bot import AgentControl

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
        environ={**VALID_ENV, "TG_USER_ID": "42", "APP_MODE": mode},
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

    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            telegram,
            telegram_user_id=42,
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

    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            telegram,
            telegram_user_id=42,
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

    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            telegram,
            telegram_user_id=42,
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