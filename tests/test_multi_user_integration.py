"""
Stage 3.8 — Integration multi-user regression test.

Verifies that two different Telegram users can work with the system
independently, without cross-contamination of data.

Scenario:
    User 42 → profile → resume → vacancy → approval → application
    User 43 → profile → resume → vacancy → approval → application
    Cross-user → attempt to use other user's data → REJECTED
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from approval import ApplicationPermission, ApprovalGuard
from config import load_settings
from database import Database, VacancyStatus
from hh_client import HHClient

# Re-use existing test helpers from test_hh_client.py
from tests.test_hh_client import (
    FakeContext,
    ResumeTrackingPage,
)

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)

# Minimal environment for Settings — no global profile.yaml
_MINIMAL_ENV = {
    "TG_BOT_TOKEN": "fake-token",
    "TG_USER_ID": "42",
    "APP_MODE": "approval",
    "ENABLE_REAL_APPLY": "true",
    "BROWSER_BACKEND": "playwright",
    "BROWSER_HEADLESS": "false",
}


def _make_settings(tmp_path: Path) -> object:
    """Create Settings with NO global profile.yaml (profile_path=None)."""
    loaded = load_settings(
        profile_path=None,
        environ=_MINIMAL_ENV,
    )
    return replace(
        loaded,
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
        max_applications_per_day=10,
        approval_ttl_minutes=30,
    )


def _setup_user_with_profile_and_resume(
    database: Database,
    telegram_user_id: int,
    resume_title: str,
    now: datetime,
) -> None:
    """Create user, user profile, search profile, and active resume."""
    database.get_or_create_user(
        telegram_user_id=telegram_user_id,
        username=f"user{telegram_user_id}",
        first_name=f"User {telegram_user_id}",
        now=now,
    )
    database.create_user_profile(
        telegram_user_id=telegram_user_id,
        name=f"Name {telegram_user_id}",
        location="Almaty" if telegram_user_id == 42 else "Astana",
        experience_summary="Experience",
        education="Bachelor",
        now=now,
    )
    database.create_search_profile(
        telegram_user_id=telegram_user_id,
        name=f"Profile {telegram_user_id}",
        search_queries=["Python"],
        areas=["160"],
        experience_filters=["between1And3"],
        desired_positions=[f"Position {telegram_user_id}"],
        technologies=["Python"],
        work_format=["Remote"],
        enabled=True,
        now=now,
    )
    database.create_resume(
        telegram_user_id=telegram_user_id,
        hh_resume_id=f"resume-{telegram_user_id}",
        title=resume_title,
        is_active=True,
        now=now,
    )


def _setup_vacancy_and_approve(
    database: Database,
    job_id: str,
    telegram_user_id: int,
    now: datetime,
) -> str:
    """Discover vacancy, request approval, and approve it."""
    database.discover(
        job_id=job_id,
        title=f"Job {job_id}",
        company="Test Co",
        url=f"https://example.com/vacancy/{job_id}",
        description_hash="hash",
        search_query="Python",
        discovered_at=now,
    )
    database.request_approval(job_id=job_id, now=now)
    permit = database.approve(
        job_id=job_id,
        telegram_user_id=telegram_user_id,
        expected_user_id=telegram_user_id,
        now=now + timedelta(minutes=1),
        ttl_minutes=30,
    )
    assert permit is not None
    return permit


# ============================================================================
# Positive scenarios: each user has their own isolated flow
# ============================================================================

def test_user_42_full_flow_isolated(tmp_path: Path) -> None:
    """User 42: profile → resume → vacancy → approval → application."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    now = NOW

    # 1. Setup user 42 with profile and resume
    _setup_user_with_profile_and_resume(database, 42, "Resume FortyTwo", now)

    # 2. Verify profile is assembled from DB (no global YAML)
    profile = database.get_active_profile(42)
    assert profile is not None
    assert profile.candidate.name == "Name 42"
    assert profile.hh.resume_name == "Resume FortyTwo"

    # 3. Setup vacancy and get approval permit
    permit_42 = _setup_vacancy_and_approve(database, "job-42", 42, now)

    # 4. Verify vacancy state before submission
    vacancy = database.get("job-42", telegram_user_id=42)
    assert vacancy is not None
    assert vacancy.status is VacancyStatus.APPROVED
    assert vacancy.approver_id == 42

    # 5. Submit application via HHClient with fake browser.
    #    NOTE: submit_application() calls approval_guard.claim() internally,
    #    so we do NOT call guard.claim() separately here.
    guard = ApprovalGuard(settings, database, now_factory=lambda: now + timedelta(minutes=2))
    permission = ApplicationPermission("job-42", permit_42, 42)

    page = ResumeTrackingPage(expected_resume="Resume FortyTwo")
    client = HHClient(
        FakeContext(page),
        settings,
        database,
        guard,
        sleep=lambda _: asyncio.sleep(0),
        now_factory=lambda: now + timedelta(minutes=3),
    )
    result = asyncio.run(client.submit_application(permission))

    # 6. Assertions — full flow completed
    assert result is True
    assert "selected_resume:Resume FortyTwo" in page.actions
    assert database.get("job-42").status is VacancyStatus.APPLIED


def test_user_43_full_flow_isolated(tmp_path: Path) -> None:
    """User 43: profile → resume → vacancy → approval → application."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    now = NOW

    # 1. Setup user 43
    _setup_user_with_profile_and_resume(database, 43, "Resume FortyThree", now)

    # 2. Verify profile
    profile = database.get_active_profile(43)
    assert profile is not None
    assert profile.candidate.name == "Name 43"
    assert profile.hh.resume_name == "Resume FortyThree"

    # 3. Setup vacancy and approve
    permit_43 = _setup_vacancy_and_approve(database, "job-43", 43, now)

    # 4. Verify vacancy ownership
    vacancy = database.get("job-43", telegram_user_id=43)
    assert vacancy is not None
    assert vacancy.status is VacancyStatus.APPROVED
    assert vacancy.approver_id == 43

    # 5. Submit via HHClient (claim happens inside submit_application)
    guard = ApprovalGuard(settings, database, now_factory=lambda: now + timedelta(minutes=2))
    permission = ApplicationPermission("job-43", permit_43, 43)

    page = ResumeTrackingPage(expected_resume="Resume FortyThree")
    client = HHClient(
        FakeContext(page),
        settings,
        database,
        guard,
        sleep=lambda _: asyncio.sleep(0),
        now_factory=lambda: now + timedelta(minutes=3),
    )
    result = asyncio.run(client.submit_application(permission))

    # 6. Assertions
    assert result is True
    assert "selected_resume:Resume FortyThree" in page.actions
    assert database.get("job-43").status is VacancyStatus.APPLIED


# ============================================================================
# Cross-user isolation: negative scenarios
# ============================================================================

def test_user_42_cannot_claim_user_43_permit(tmp_path: Path) -> None:
    """User 42 cannot use User 43's approval permit."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    now = NOW

    # Setup both users
    _setup_user_with_profile_and_resume(database, 42, "Resume FortyTwo", now)
    _setup_user_with_profile_and_resume(database, 43, "Resume FortyThree", now)

    # User 43 gets approval for job-43
    permit_43 = _setup_vacancy_and_approve(database, "job-43", 43, now)

    # User 42 tries to claim with User 43's permit
    guard = ApprovalGuard(settings, database, now_factory=lambda: now + timedelta(minutes=2))
    permission = ApplicationPermission("job-43", permit_43, 42)
    claim = guard.claim(permission)

    assert not claim.allowed
    assert claim.reason == "wrong_user"
    assert database.get("job-43").status is VacancyStatus.APPROVED  # unchanged


def test_user_43_cannot_claim_user_42_permit(tmp_path: Path) -> None:
    """User 43 cannot use User 42's approval permit."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    now = NOW

    _setup_user_with_profile_and_resume(database, 42, "Resume FortyTwo", now)
    _setup_user_with_profile_and_resume(database, 43, "Resume FortyThree", now)

    permit_42 = _setup_vacancy_and_approve(database, "job-42", 42, now)

    guard = ApprovalGuard(settings, database, now_factory=lambda: now + timedelta(minutes=2))
    permission = ApplicationPermission("job-42", permit_42, 43)
    claim = guard.claim(permission)

    assert not claim.allowed
    assert claim.reason == "wrong_user"
    assert database.get("job-42").status is VacancyStatus.APPROVED


def test_user_42_cannot_use_user_43_resume(tmp_path: Path) -> None:
    """User 42's application must NOT select User 43's resume."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    now = NOW

    _setup_user_with_profile_and_resume(database, 42, "Resume FortyTwo", now)
    _setup_user_with_profile_and_resume(database, 43, "Resume FortyThree", now)

    permit_42 = _setup_vacancy_and_approve(database, "job-42", 42, now)

    # The fake page only shows User 43's resume (simulating wrong resume on HH page)
    page = ResumeTrackingPage(expected_resume="Resume FortyThree")
    guard = ApprovalGuard(settings, database, now_factory=lambda: now + timedelta(minutes=2))
    client = HHClient(
        FakeContext(page),
        settings,
        database,
        guard,
        sleep=lambda _: asyncio.sleep(0),
        now_factory=lambda: now + timedelta(minutes=3),
    )

    permission = ApplicationPermission("job-42", permit_42, 42)
    result = asyncio.run(client.submit_application(permission))

    # Should fail because user 42's resume is not found on the page
    assert result is False
    assert "selected_resume:Resume FortyThree" not in page.actions


def test_user_43_cannot_use_user_42_resume(tmp_path: Path) -> None:
    """User 43's application must NOT select User 42's resume."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    now = NOW

    _setup_user_with_profile_and_resume(database, 42, "Resume FortyTwo", now)
    _setup_user_with_profile_and_resume(database, 43, "Resume FortyThree", now)

    permit_43 = _setup_vacancy_and_approve(database, "job-43", 43, now)

    # Fake page only shows User 42's resume
    page = ResumeTrackingPage(expected_resume="Resume FortyTwo")
    guard = ApprovalGuard(settings, database, now_factory=lambda: now + timedelta(minutes=2))
    client = HHClient(
        FakeContext(page),
        settings,
        database,
        guard,
        sleep=lambda _: asyncio.sleep(0),
        now_factory=lambda: now + timedelta(minutes=3),
    )

    permission = ApplicationPermission("job-43", permit_43, 43)
    result = asyncio.run(client.submit_application(permission))

    assert result is False
    assert "selected_resume:Resume FortyTwo" not in page.actions


# ============================================================================
# Full parallel scenario: both users complete their flows independently
# ============================================================================

def test_both_users_complete_flows_independently(tmp_path: Path) -> None:
    """Both users complete full flows; data does not mix."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    now = NOW

    # Setup both users with full profiles
    _setup_user_with_profile_and_resume(database, 42, "Resume FortyTwo", now)
    _setup_user_with_profile_and_resume(database, 43, "Resume FortyThree", now)

    # Verify profile isolation
    profile_42 = database.get_active_profile(42)
    profile_43 = database.get_active_profile(43)
    assert profile_42.candidate.name == "Name 42"
    assert profile_42.hh.resume_name == "Resume FortyTwo"
    assert profile_43.candidate.name == "Name 43"
    assert profile_43.hh.resume_name == "Resume FortyThree"

    # Setup vacancies
    permit_42 = _setup_vacancy_and_approve(database, "job-42", 42, now)
    permit_43 = _setup_vacancy_and_approve(database, "job-43", 43, now)

    # Verify vacancy isolation
    assert database.get("job-42").approver_id == 42
    assert database.get("job-43").approver_id == 43

    # Submit both via HHClient (claim happens inside submit_application)
    guard = ApprovalGuard(settings, database, now_factory=lambda: now + timedelta(minutes=2))

    page_42 = ResumeTrackingPage(expected_resume="Resume FortyTwo")
    page_43 = ResumeTrackingPage(expected_resume="Resume FortyThree")

    client_42 = HHClient(
        FakeContext(page_42),
        settings,
        database,
        guard,
        sleep=lambda _: asyncio.sleep(0),
        now_factory=lambda: now + timedelta(minutes=3),
    )
    client_43 = HHClient(
        FakeContext(page_43),
        settings,
        database,
        guard,
        sleep=lambda _: asyncio.sleep(0),
        now_factory=lambda: now + timedelta(minutes=3),
    )

    result_42 = asyncio.run(client_42.submit_application(
        ApplicationPermission("job-42", permit_42, 42)
    ))
    result_43 = asyncio.run(client_43.submit_application(
        ApplicationPermission("job-43", permit_43, 43)
    ))

    assert result_42 is True
    assert result_43 is True
    assert "selected_resume:Resume FortyTwo" in page_42.actions
    assert "selected_resume:Resume FortyThree" in page_43.actions
    assert database.get("job-42").status is VacancyStatus.APPLIED
    assert database.get("job-43").status is VacancyStatus.APPLIED


# ============================================================================
# Profile isolation: no global YAML, profiles from DB only
# ============================================================================

def test_profiles_work_without_global_yaml(tmp_path: Path) -> None:
    """User profiles are assembled from DB when no global profile.yaml exists."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    now = NOW

    # Settings has empty global profile (no YAML loaded)
    assert settings.profile.candidate.name == ""
    assert settings.profile.hh.resume_name == ""

    # Create users with DB profiles
    _setup_user_with_profile_and_resume(database, 42, "Resume FortyTwo", now)
    _setup_user_with_profile_and_resume(database, 43, "Resume FortyThree", now)

    # Profiles come from DB, not global YAML
    profile_42 = database.get_active_profile(42)
    profile_43 = database.get_active_profile(43)

    assert profile_42 is not None
    assert profile_42.candidate.name == "Name 42"
    assert profile_42.hh.resume_name == "Resume FortyTwo"
    assert profile_42.hh.search_queries == ("Python",)

    assert profile_43 is not None
    assert profile_43.candidate.name == "Name 43"
    assert profile_43.hh.resume_name == "Resume FortyThree"
    assert profile_43.hh.areas == ("160",)

    # Cross-check: user 42 profile does not contain user 43 data
    assert profile_42.candidate.name != "Name 43"
    assert profile_42.hh.resume_name != "Resume FortyThree"


# ============================================================================
# Permit reuse and cross-permit attacks
# ============================================================================

def test_user_42_permit_cannot_be_reused_by_user_43(tmp_path: Path) -> None:
    """User 43 cannot use User 42's permit even after User 42 has claimed it."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    now = NOW

    _setup_user_with_profile_and_resume(database, 42, "Resume FortyTwo", now)
    _setup_user_with_profile_and_resume(database, 43, "Resume FortyThree", now)

    permit_42 = _setup_vacancy_and_approve(database, "job-42", 42, now)

    # User 42 claims successfully
    guard = ApprovalGuard(settings, database, now_factory=lambda: now + timedelta(minutes=2))
    claim_42 = guard.claim(ApplicationPermission("job-42", permit_42, 42))
    assert claim_42.allowed

    # User 43 tries to use the same permit
    claim_43 = guard.claim(ApplicationPermission("job-42", permit_42, 43))
    assert not claim_43.allowed
    assert claim_43.reason in ("wrong_user", "invalid_permit", "concurrent_claim", "invalid_status")


def test_user_42_cannot_approve_user_43_vacancy(tmp_path: Path) -> None:
    """User 42 cannot approve (click approve on) User 43's vacancy."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    now = NOW

    _setup_user_with_profile_and_resume(database, 42, "Resume FortyTwo", now)
    _setup_user_with_profile_and_resume(database, 43, "Resume FortyThree", now)

    # User 43's vacancy goes to PENDING_APPROVAL
    database.discover(
        job_id="job-43",
        title="Job 43",
        company="Test Co",
        url="https://example.com/vacancy/job-43",
        description_hash="hash",
        search_query="Python",
        discovered_at=now,
    )
    database.request_approval(job_id="job-43", now=now)

    # User 42 tries to approve it
    permit = database.approve(
        job_id="job-43",
        telegram_user_id=42,
        expected_user_id=43,  # expected is 43, but caller is 42
        now=now + timedelta(minutes=1),
    )
    assert permit is None
    assert database.get("job-43").status is VacancyStatus.PENDING_APPROVAL