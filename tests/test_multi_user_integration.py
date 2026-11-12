"""Stage 3.8 multi-user isolation integration tests.

Verifies that per-user profiles, vacancies, approvals, resumes, daily limits,
runtime contexts, pause/wake events, circuit-breaker state, and Telegram
notification routing remain strictly isolated across telegram_user_id values.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from approval import ApplicationPermission, ApprovalGuard, ApprovalService
from config import load_settings, load_user_settings
from database import Database, VacancyStatus
from hh_client import HHClient
from runtime_context import HHAccountOwnershipError, create_runtime_context

# Re-use existing test helpers from test_hh_client.py
from tests.test_hh_client import (
    FakeContext,
    ResumeTrackingPage,
)

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)

# Minimal environment for Settings — no global profile.yaml, no TG_USER_ID
_MINIMAL_ENV = {
    "TG_BOT_TOKEN": "fake-token",
    "APP_MODE": "dry_run",
    "ENABLE_REAL_APPLY": "false",
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
        telegram_user_id=telegram_user_id,
    )
    database.request_approval(job_id=job_id, now=now, telegram_user_id=telegram_user_id)
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
# Stage 4\5: HHClient user identity isolation
# ============================================================================

def test_hh_client_user_identity_isolation(tmp_path: Path) -> None:
    """HHClient для user 42 и user 43 имеют разные telegram_user_id."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    guard = ApprovalGuard(settings, database)

    us_42 = load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30)
    us_43 = load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30)
    client_42 = HHClient(
        FakeContext(ResumeTrackingPage()),
        settings,
        database,
        guard,
        telegram_user_id=42,
        user_settings=us_42,
    )
    client_43 = HHClient(
        FakeContext(ResumeTrackingPage()),
        settings,
        database,
        guard,
        telegram_user_id=43,
        user_settings=us_43,
    )

    assert client_42.telegram_user_id == 42
    assert client_43.telegram_user_id == 43
    assert client_42.telegram_user_id != client_43.telegram_user_id


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
    guard = ApprovalGuard(load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30), database, now_factory=lambda: now + timedelta(minutes=2))
    permission = ApplicationPermission("job-42", permit_42, 42)

    page = ResumeTrackingPage(expected_resume="Resume FortyTwo")
    client = HHClient(
        FakeContext(page),
        settings,
        database,
        guard,
        telegram_user_id=42,
        user_settings=load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30),
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
    guard = ApprovalGuard(load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30), database, now_factory=lambda: now + timedelta(minutes=2))
    permission = ApplicationPermission("job-43", permit_43, 43)

    page = ResumeTrackingPage(expected_resume="Resume FortyThree")
    client = HHClient(
        FakeContext(page),
        settings,
        database,
        guard,
        telegram_user_id=43,
        user_settings=load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30),
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
    guard = ApprovalGuard(load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30), database, now_factory=lambda: now + timedelta(minutes=2))
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

    guard = ApprovalGuard(load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30), database, now_factory=lambda: now + timedelta(minutes=2))
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
    guard = ApprovalGuard(load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30), database, now_factory=lambda: now + timedelta(minutes=2))
    client = HHClient(
        FakeContext(page),
        settings,
        database,
        guard,
        telegram_user_id=42,
        user_settings=load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30),
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
    guard = ApprovalGuard(load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30), database, now_factory=lambda: now + timedelta(minutes=2))
    client = HHClient(
        FakeContext(page),
        settings,
        database,
        guard,
        telegram_user_id=43,
        user_settings=load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30),
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
    guard = ApprovalGuard(load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30), database, now_factory=lambda: now + timedelta(minutes=2))

    page_42 = ResumeTrackingPage(expected_resume="Resume FortyTwo")
    page_43 = ResumeTrackingPage(expected_resume="Resume FortyThree")

    us_42 = load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30)
    us_43 = load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30)
    client_42 = HHClient(
        FakeContext(page_42),
        settings,
        database,
        guard,
        telegram_user_id=42,
        user_settings=us_42,
        sleep=lambda _: asyncio.sleep(0),
        now_factory=lambda: now + timedelta(minutes=3),
    )
    client_43 = HHClient(
        FakeContext(page_43),
        settings,
        database,
        guard,
        telegram_user_id=43,
        user_settings=us_43,
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
    guard = ApprovalGuard(load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30), database, now_factory=lambda: now + timedelta(minutes=2))
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


# ============================================================================
# ApprovalService isolation
# ============================================================================

def test_approval_service_user_42_cannot_apply_for_user_43(tmp_path: Path) -> None:
    """ApprovalService.enforce user isolation: User 42 cannot trigger apply for User 43's vacancy."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    now = NOW

    _setup_user_with_profile_and_resume(database, 42, "Resume FortyTwo", now)
    _setup_user_with_profile_and_resume(database, 43, "Resume FortyThree", now)

    # User 43's vacancy
    database.discover(
        job_id="job-43",
        title="Job 43",
        company="Test Co",
        url="https://example.com/vacancy/job-43",
        description_hash="hash",
        search_query="Python",
        discovered_at=now,
        telegram_user_id=43,
    )
    database.request_approval(job_id="job-43", now=now, telegram_user_id=43)

    # User 42 tries to approve_and_apply for User 43's vacancy
    guard = ApprovalGuard(load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30), database, now_factory=lambda: now + timedelta(minutes=2))
    page = ResumeTrackingPage(expected_resume="Resume FortyThree")
    client = HHClient(
        FakeContext(page),
        settings,
        database,
        guard,
        telegram_user_id=42,
        user_settings=load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30),
        sleep=lambda _: asyncio.sleep(0),
        now_factory=lambda: now + timedelta(minutes=3),
    )
    service = ApprovalService(
        load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30),
        database,
        client,
        now_factory=lambda: now + timedelta(minutes=1),
    )

    result = asyncio.run(service.approve_and_apply("job-43", 42))

    assert not result.ok
    assert database.get("job-43").status is VacancyStatus.PENDING_APPROVAL
    assert database.get("job-43").approver_id is None
    assert database.get("job-43").approved_at is None


# ============================================================================
# Database-level isolation: list operations
# ============================================================================

def test_user_vacancies_are_isolated_in_list_operations(tmp_path: Path) -> None:
    """User-scoped list operations return only owned data."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    now = NOW

    _setup_user_with_profile_and_resume(database, 42, "Resume FortyTwo", now)
    _setup_user_with_profile_and_resume(database, 43, "Resume FortyThree", now)

    # User 42 has a vacancy
    database.discover(
        job_id="job-42",
        title="Job 42",
        company="Test Co",
        url="https://example.com/vacancy/job-42",
        description_hash="hash",
        search_query="Python",
        discovered_at=now,
        telegram_user_id=42,
    )
    database.request_approval(job_id="job-42", now=now, telegram_user_id=42)

    # User 43 has a vacancy
    database.discover(
        job_id="job-43",
        title="Job 43",
        company="Test Co",
        url="https://example.com/vacancy/job-43",
        description_hash="hash",
        search_query="Python",
        discovered_at=now,
        telegram_user_id=43,
    )
    database.request_approval(job_id="job-43", now=now, telegram_user_id=43)

    # pending() is user-scoped
    pending_42 = database.pending(telegram_user_id=42)
    pending_43 = database.pending(telegram_user_id=43)

    assert len(pending_42) == 1
    assert pending_42[0].id == "job-42"
    assert len(pending_43) == 1
    assert pending_43[0].id == "job-43"

    # stats() is user-scoped
    stats_42 = database.stats(telegram_user_id=42)
    stats_43 = database.stats(telegram_user_id=43)

    assert stats_42["pending_approval"] == 1
    assert stats_43["pending_approval"] == 1
    # Each user sees only their own vacancy
    assert stats_42.get("discovered", 0) == 0
    assert stats_43.get("discovered", 0) == 0


# ============================================================================
# Daily limit isolation
# ============================================================================

def test_daily_limit_is_user_scoped(tmp_path: Path) -> None:
    """Daily application limit is isolated per user."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    now = NOW

    _setup_user_with_profile_and_resume(database, 42, "Resume FortyTwo", now)
    _setup_user_with_profile_and_resume(database, 43, "Resume FortyThree", now)

    # User 42 applies to a vacancy (completes full flow)
    permit_42 = _setup_vacancy_and_approve(database, "job-42", 42, now)
    guard = ApprovalGuard(load_user_settings(settings, app_mode="approval", enable_real_apply=True, max_applications_per_day=10, approval_ttl_minutes=30), database, now_factory=lambda: now + timedelta(minutes=2))
    claim_42 = guard.claim(ApplicationPermission("job-42", permit_42, 42))
    assert claim_42.allowed
    database.complete_application("job-42", permit_42, success=True, now=now + timedelta(minutes=3))

    # User 43 should have their own independent daily limit
    assert database.applied_today(now + timedelta(minutes=4), telegram_user_id=43) == 0
    assert database.applied_today(now + timedelta(minutes=4), telegram_user_id=42) == 1

# ============================================================================
# Stage 4\6 — per-user settings identity and isolation
# ============================================================================

def test_users_have_distinct_user_settings_instances(tmp_path: Path) -> None:
    """User 42 and user 43 hold different UserSettings instances."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    us_42 = load_user_settings(settings, max_applications_per_day=10)
    us_43 = load_user_settings(settings, max_applications_per_day=10)

    ctx_42 = create_runtime_context(42, settings, database, us_42)
    ctx_43 = create_runtime_context(43, settings, database, us_43)

    assert us_42 is not us_43
    assert ctx_42.user_settings is us_42
    assert ctx_43.user_settings is us_43


def test_user_42_replace_does_not_affect_user_43(tmp_path: Path) -> None:
    """Stage 4\6 acceptance: replace() for user 42 leaves user 43 intact."""
    from dataclasses import replace as dc_replace

    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    ctx_42 = create_runtime_context(
        42, settings, database,
        load_user_settings(settings, max_applications_per_day=10),
    )
    ctx_43 = create_runtime_context(
        43, settings, database,
        load_user_settings(settings, max_applications_per_day=10),
    )

    ctx_42.user_settings = dc_replace(
        ctx_42.user_settings,
        app_mode="approval",
        enable_real_apply=True,
        max_applications_per_day=99,
        check_interval_minutes=7,
    )

    assert ctx_42.user_settings.app_mode == "approval"
    assert ctx_42.user_settings.enable_real_apply is True
    assert ctx_42.user_settings.max_applications_per_day == 99
    assert ctx_42.user_settings.check_interval_minutes == 7

    assert ctx_43.user_settings.app_mode == "dry_run"
    assert ctx_43.user_settings.enable_real_apply is False
    assert ctx_43.user_settings.max_applications_per_day == 10
    assert ctx_43.user_settings.check_interval_minutes == 30

# ============================================================================
# Phase 4.8 — multi-user runtime isolation integration tests
# ============================================================================

from hh_client import (
    CompanyDetails,
    PageState,
    VacancyDetails,
    VacancySearchResult,
    VacancySummary,
)
from main import agent_loop, run_search_cycle
from tg_bot import TelegramService


class _Phase48FakeTelegram:
    """Duck-typed Telegram surface used by run_search_cycle / agent_loop."""

    def __init__(self) -> None:
        self.previews: list[tuple[str, bool, int | None]] = []
        self.notifications: list[tuple[str, int | None]] = []

    async def send_preview(
        self, vacancy, include_actions: bool, telegram_user_id: int | None = None
    ):
        self.previews.append((vacancy.id, include_actions, telegram_user_id))

    async def notify(self, text: str, telegram_user_id: int | None = None):
        self.notifications.append((text, telegram_user_id))

    async def request_captcha(
        self, screenshot, title: str, timeout_seconds: int, telegram_user_id: int | None = None
    ):
        return None


class _Phase48FakeBot:
    """Records every bot.send_message call for chat_id routing assertions."""

    def __init__(self) -> None:
        self.messages: list[tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str, **kwargs):
        self.messages.append((chat_id, text))


class _Phase48SuccessHHClient:
    """Completes a search cycle with zero vacancies; counts calls per instance."""

    def __init__(self) -> None:
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


class _Phase48ErroringHHClient:
    """Always returns a search error (triggers consecutive_search_errors)."""

    def __init__(self, error_reason: str = "network_error") -> None:
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



class _Phase48ErroringBarrierHHClient:
    """Erroring HHClient that parks on the *first* search_vacancies call only.

    Subsequent calls return the error immediately so three consecutive failures
    can still trip the circuit breaker after the concurrent barrier is released.
    """

    def __init__(
        self,
        *,
        entered: asyncio.Event,
        release: asyncio.Event,
        error_reason: str = "network_error",
    ) -> None:
        self.entered = entered
        self.release = release
        self.error_reason = error_reason
        self.search_calls = 0
        self._barrier_pending = True

    async def search_vacancies(self, query, areas, experience_filters):
        self.search_calls += 1
        if self._barrier_pending:
            self._barrier_pending = False
            self.entered.set()
            await self.release.wait()
        return VacancySearchResult([], 0, 0, error_reason=self.error_reason)

    async def read_vacancy(self, summary, captcha_solver=None):
        return VacancyDetails(summary, PageState.NETWORK_ERROR, error="read failed")

    async def check_messages(self, notify):
        pass

    async def read_company_details(self, company_url: str):
        return CompanyDetails(None, None)


class _Phase48SuccessBarrierHHClient:
    """Success HHClient that parks inside search_vacancies until release fires."""

    def __init__(self, *, entered: asyncio.Event, release: asyncio.Event) -> None:
        self.entered = entered
        self.release = release
        self.search_calls = 0

    async def search_vacancies(self, query, areas, experience_filters):
        self.search_calls += 1
        self.entered.set()
        await self.release.wait()
        return VacancySearchResult([], 0, 0)

    async def read_vacancy(self, summary, captcha_solver=None):
        return VacancyDetails(summary, PageState.VACANCY_LOADED, "Co", "desc")

    async def check_messages(self, notify):
        pass

    async def read_company_details(self, company_url: str):
        return CompanyDetails(None, None)

class _Phase48BarrierHHClient:
    """HHClient that parks inside search_vacancies until a shared release Event fires.

    ``entered`` is set when the coroutine reaches the search call so a test can
    wait until both users are simultaneously blocked at the barrier before
    releasing them.
    """

    def __init__(
        self,
        *,
        entered: asyncio.Event,
        release: asyncio.Event,
        summaries: list[VacancySummary],
    ) -> None:
        self.entered = entered
        self.release = release
        self.summaries = summaries
        self.search_calls = 0

    async def search_vacancies(self, query, areas, experience_filters):
        self.search_calls += 1
        self.entered.set()
        await self.release.wait()
        return VacancySearchResult(
            list(self.summaries),
            found_results=len(self.summaries),
            duplicates=0,
        )

    async def read_vacancy(self, summary, captcha_solver=None):
        return VacancyDetails(
            summary,
            PageState.VACANCY_LOADED,
            company=f"Company-{summary.id}",
            description=f"Description for {summary.id}",
        )

    async def check_messages(self, notify):
        pass

    async def read_company_details(self, company_url: str):
        return CompanyDetails(None, None)


class _Phase48NullSender:
    async def submit_application(self, permission) -> bool:
        return False


def _phase48_settings(tmp_path: Path):
    """Settings with a profile that has at least one search query."""
    from tests.test_config import VALID_ENV, write_profile

    loaded = load_settings(
        profile_path=write_profile(tmp_path),
        environ={**VALID_ENV, "APP_MODE": "dry_run", "ENABLE_REAL_APPLY": "false"},
    )
    return replace(
        loaded,
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )


def _phase48_two_contexts(tmp_path: Path, *, check_interval_minutes: int = 1):
    """Two independent runtime contexts with non-empty search queries."""
    app_settings = _phase48_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    database.create_user(telegram_user_id=42, username="user42", now=NOW)
    database.create_user(telegram_user_id=43, username="user43", now=NOW)

    us_42 = load_user_settings(
        app_settings, check_interval_minutes=check_interval_minutes
    )
    us_43 = load_user_settings(
        app_settings, check_interval_minutes=check_interval_minutes
    )
    ctx_42 = create_runtime_context(42, app_settings, database, us_42)
    ctx_43 = create_runtime_context(43, app_settings, database, us_43)
    return app_settings, database, ctx_42, ctx_43


def test_concurrent_runtime_contexts_isolation(tmp_path: Path) -> None:
    """Phase 4.8: two agent_loops hit search concurrently behind a shared barrier.

    Both HH clients park inside ``search_vacancies`` and set their own
    ``entered`` Event. The test waits until *both* events are set (proving the
    two searches overlap in time), then releases them. Each user gets a
    distinct vacancy id; database rows must not cross-contaminate.
    """
    _, database, ctx_42, ctx_43 = _phase48_two_contexts(
        tmp_path, check_interval_minutes=60
    )
    assert ctx_42.wake_event is not ctx_43.wake_event

    entered_42 = asyncio.Event()
    entered_43 = asyncio.Event()
    release = asyncio.Event()

    summary_42 = VacancySummary(
        "job-only-42",
        "Python role for user 42",
        "https://example.com/vacancy/job-only-42",
        "Python",
    )
    summary_43 = VacancySummary(
        "job-only-43",
        "Python role for user 43",
        "https://example.com/vacancy/job-only-43",
        "Python",
    )

    hh_42 = _Phase48BarrierHHClient(
        entered=entered_42, release=release, summaries=[summary_42]
    )
    hh_43 = _Phase48BarrierHHClient(
        entered=entered_43, release=release, summaries=[summary_43]
    )
    tg_42 = _Phase48FakeTelegram()
    tg_43 = _Phase48FakeTelegram()
    preview_42_done = asyncio.Event()
    preview_43_done = asyncio.Event()

    orig_preview_42 = tg_42.send_preview
    orig_preview_43 = tg_43.send_preview

    async def tracked_preview_42(vacancy, include_actions: bool, telegram_user_id: int | None = None):
        await orig_preview_42(vacancy, include_actions, telegram_user_id)
        if vacancy.id == "job-only-42":
            preview_42_done.set()

    async def tracked_preview_43(vacancy, include_actions: bool, telegram_user_id: int | None = None):
        await orig_preview_43(vacancy, include_actions, telegram_user_id)
        if vacancy.id == "job-only-43":
            preview_43_done.set()

    tg_42.send_preview = tracked_preview_42  # type: ignore[method-assign]
    tg_43.send_preview = tracked_preview_43  # type: ignore[method-assign]

    async def scenario() -> None:
        task_42 = asyncio.create_task(agent_loop(ctx_42, hh_42, tg_42))
        task_43 = asyncio.create_task(agent_loop(ctx_43, hh_43, tg_43))
        try:
            # Strict barrier: both users must be inside search_vacancies at once.
            await asyncio.wait_for(
                asyncio.gather(entered_42.wait(), entered_43.wait()),
                timeout=5.0,
            )
            assert hh_42.search_calls >= 1
            assert hh_43.search_calls >= 1
            # Neither has returned from search yet — still concurrent.
            assert not release.is_set()

            release.set()

            # Wait until each user finished processing their distinct vacancy.
            await asyncio.wait_for(
                asyncio.gather(preview_42_done.wait(), preview_43_done.wait()),
                timeout=5.0,
            )

            ctx_42.paused = True
            ctx_43.paused = True
            ctx_42.wake_event.set()
            ctx_43.wake_event.set()
        finally:
            for t in (task_42, task_43):
                if not t.done():
                    t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

    asyncio.run(scenario())

    assert hh_42.search_calls >= 1
    assert hh_43.search_calls >= 1

    vac_42 = database.get("job-only-42", telegram_user_id=42)
    vac_43 = database.get("job-only-43", telegram_user_id=43)
    assert vac_42 is not None
    assert vac_43 is not None
    # Cross-user lookup must not return the other user's vacancy.
    assert database.get("job-only-42", telegram_user_id=43) is None
    assert database.get("job-only-43", telegram_user_id=42) is None

    run_42 = database.latest_search_run(telegram_user_id=42)
    run_43 = database.latest_search_run(telegram_user_id=43)
    assert run_42 is not None and run_42.telegram_user_id == 42
    assert run_43 is not None and run_43.telegram_user_id == 43
    assert run_42.state == "completed"
    assert run_43.state == "completed"
    assert run_42.new_vacancies >= 1
    assert run_43.new_vacancies >= 1

    # Previews routed to the owning user only.
    assert any(job_id == "job-only-42" and uid == 42 for job_id, _, uid in tg_42.previews)
    assert any(job_id == "job-only-43" and uid == 43 for job_id, _, uid in tg_43.previews)
    assert all(uid != 43 for _, _, uid in tg_42.previews)
    assert all(uid != 42 for _, _, uid in tg_43.previews)


def test_independent_paused_and_wake_events(tmp_path: Path) -> None:
    """Phase 4.8: pause/resume of one user does not affect the other.

    Recovery uses the real TelegramService.command('resume') path.
    """
    app_settings, database, ctx_42, ctx_43 = _phase48_two_contexts(
        tmp_path, check_interval_minutes=60
    )
    hh_42 = _Phase48SuccessHHClient()
    hh_43 = _Phase48SuccessHHClient()
    tg_42 = _Phase48FakeTelegram()
    tg_43 = _Phase48FakeTelegram()

    # Wire real TelegramService only for the resume command surface.
    approval = ApprovalService(
        load_user_settings(app_settings), database, _Phase48NullSender()
    )
    bot_service = TelegramService(
        app_settings,
        database,
        approval,
        contexts={42: ctx_42, 43: ctx_43},
        bot=object(),
    )

    cycle_43_before_resume = asyncio.Event()
    cycle_42_after_resume = asyncio.Event()
    cycle_43_after_resume = asyncio.Event()

    orig_42 = hh_42.search_vacancies
    orig_43 = hh_43.search_vacancies
    searches_43 = {"n": 0}

    async def tracked_42(*args, **kwargs):
        result = await orig_42(*args, **kwargs)
        if not ctx_42.paused:
            cycle_42_after_resume.set()
        return result

    async def tracked_43(*args, **kwargs):
        result = await orig_43(*args, **kwargs)
        searches_43["n"] += 1
        if searches_43["n"] == 1:
            cycle_43_before_resume.set()
        elif searches_43["n"] >= 2:
            cycle_43_after_resume.set()
        return result

    hh_42.search_vacancies = tracked_42  # type: ignore[method-assign]
    hh_43.search_vacancies = tracked_43  # type: ignore[method-assign]

    # Start with user 42 already paused — only 43 should run a cycle first.
    ctx_42.paused = True

    async def scenario() -> None:
        task_42 = asyncio.create_task(agent_loop(ctx_42, hh_42, tg_42))
        task_43 = asyncio.create_task(agent_loop(ctx_43, hh_43, tg_43))
        try:
            await asyncio.wait_for(cycle_43_before_resume.wait(), timeout=5.0)
            assert hh_42.search_calls == 0
            assert hh_43.search_calls >= 1
            assert ctx_42.paused is True
            assert ctx_43.paused is False

            # Real recovery path: /resume for user 42 only.
            reply = bot_service.command("resume", 42)
            assert reply == "Agent resumed."
            assert ctx_42.paused is False
            assert ctx_42.circuit_reason == ""
            assert ctx_42.consecutive_search_errors == 0
            # Force user 43's waiting loop to run another cycle so we can
            # observe independence without relying on the long interval timer.
            ctx_43.wake_event.set()

            await asyncio.wait_for(
                asyncio.gather(
                    cycle_42_after_resume.wait(),
                    cycle_43_after_resume.wait(),
                ),
                timeout=5.0,
            )

            ctx_42.paused = True
            ctx_43.paused = True
            ctx_42.wake_event.set()
            ctx_43.wake_event.set()
        finally:
            for t in (task_42, task_43):
                if not t.done():
                    t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

    asyncio.run(scenario())

    assert hh_42.search_calls >= 1
    assert hh_43.search_calls >= 2
    # Pause of 42 never flipped 43.
    assert ctx_43.circuit_reason == ""


def test_cross_user_error_and_recovery(tmp_path: Path) -> None:
    """Phase 4.8: concurrent error accumulation on 42 while 43 succeeds.

    User 42's three failing cycles and user 43's successful cycle run in the
    same event loop. Concurrency is enforced with a shared barrier: both
    users park inside ``search_vacancies`` (entered_42 / entered_43), the
    test waits until both events are set, then releases them. After the
    barrier, user 42 accumulates two more errors to trip the circuit-breaker.
    Recovery uses the real ``TelegramService.command('resume', 42)`` path.
    """
    app_settings, database, ctx_42, ctx_43 = _phase48_two_contexts(
        tmp_path, check_interval_minutes=60
    )
    entered_42 = asyncio.Event()
    entered_43 = asyncio.Event()
    release = asyncio.Event()

    err_client = _Phase48ErroringBarrierHHClient(
        entered=entered_42, release=release, error_reason="network_error"
    )
    ok_client = _Phase48SuccessBarrierHHClient(entered=entered_43, release=release)
    tg_42 = _Phase48FakeTelegram()
    tg_43 = _Phase48FakeTelegram()

    approval = ApprovalService(
        load_user_settings(app_settings), database, _Phase48NullSender()
    )
    bot_service = TelegramService(
        app_settings,
        database,
        approval,
        contexts={42: ctx_42, 43: ctx_43},
        bot=object(),
    )

    async def failing_user_42() -> None:
        # Three error cycles: the first parks at the barrier so that user 43
        # can enter search concurrently; the remaining two run after release
        # to accumulate the circuit-breaker threshold.
        for _ in range(3):
            if not ctx_42.paused:
                await run_search_cycle(
                    ctx_42, err_client, tg_42, now_factory=lambda: NOW
                )

    async def successful_user_43() -> None:
        await run_search_cycle(ctx_43, ok_client, tg_43, now_factory=lambda: NOW)

    async def scenario() -> None:
        task_42 = asyncio.create_task(failing_user_42())
        task_43 = asyncio.create_task(successful_user_43())
        try:
            # Strict barrier: both users must be inside search_vacancies together.
            await asyncio.wait_for(
                asyncio.gather(entered_42.wait(), entered_43.wait()),
                timeout=5.0,
            )
            assert err_client.search_calls >= 1
            assert ok_client.search_calls >= 1
            assert not release.is_set()

            release.set()
            await asyncio.gather(task_42, task_43)
        finally:
            for t in (task_42, task_43):
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass

    asyncio.run(scenario())

    assert ctx_42.paused is True
    assert ctx_42.circuit_reason == "search_errors"
    assert ctx_42.consecutive_search_errors >= 3

    assert ctx_43.paused is False
    assert ctx_43.circuit_reason == ""
    assert ctx_43.consecutive_search_errors == 0
    run_43 = database.latest_search_run(telegram_user_id=43)
    assert run_43 is not None
    assert run_43.state == "completed"
    assert run_43.telegram_user_id == 43

    # Real recovery for user 42.
    reply = bot_service.command("resume", 42)
    assert reply == "Agent resumed."
    assert ctx_42.paused is False
    assert ctx_42.circuit_reason == ""
    assert ctx_42.consecutive_search_errors == 0

    # After recovery, user 42 completes a successful cycle with a healthy client.
    # Use a non-barrier success client so the post-recovery cycle is not blocked.
    healthy_client = _Phase48SuccessHHClient()
    run_42_ok = asyncio.run(
        run_search_cycle(ctx_42, healthy_client, tg_42, now_factory=lambda: NOW)
    )
    assert run_42_ok.state == "completed"
    assert ctx_42.paused is False
    assert ctx_42.consecutive_search_errors == 0

    # User 43 remains unaffected throughout.
    assert ctx_43.paused is False
    assert ctx_43.circuit_reason == ""
    assert ctx_43.consecutive_search_errors == 0


def test_database_and_notification_routing_isolation(tmp_path: Path) -> None:
    """Phase 4.8: search_runs and real bot.send_message chat_id stay per-user.

    Uses a single real ``TelegramService`` backed by ``_Phase48FakeBot`` so
    that ``notify(..., telegram_user_id=N)`` is observed as
    ``bot.send_message(chat_id=N, ...)``.

    Concurrency is enforced with a shared barrier: the first erroring search of
    user 42 and the successful search of user 43 both park inside
    ``search_vacancies``. The test proceeds only after *both* ``entered`` Events
    are set, proving the searches overlap, then releases them.
    """
    app_settings, database, ctx_42, ctx_43 = _phase48_two_contexts(
        tmp_path, check_interval_minutes=60
    )

    entered_42 = asyncio.Event()
    entered_43 = asyncio.Event()
    release = asyncio.Event()

    err_client = _Phase48ErroringBarrierHHClient(
        entered=entered_42, release=release, error_reason="network_error"
    )
    ok_client = _Phase48SuccessBarrierHHClient(entered=entered_43, release=release)

    fake_bot = _Phase48FakeBot()
    approval = ApprovalService(
        load_user_settings(app_settings), database, _Phase48NullSender()
    )
    # One shared TelegramService — production routing path via chat_id.
    telegram = TelegramService(
        app_settings,
        database,
        approval,
        contexts={42: ctx_42, 43: ctx_43},
        bot=fake_bot,
    )

    async def scenario() -> None:
        async def fail_42() -> None:
            # Three error cycles: the first one blocks at the barrier so that
            # user 43 can enter search concurrently; the remaining two run
            # after release to accumulate the circuit-breaker threshold.
            for _ in range(3):
                if not ctx_42.paused:
                    await run_search_cycle(
                        ctx_42, err_client, telegram, now_factory=lambda: NOW
                    )

        async def ok_43() -> None:
            await run_search_cycle(
                ctx_43, ok_client, telegram, now_factory=lambda: NOW
            )

        task_42 = asyncio.create_task(fail_42())
        task_43 = asyncio.create_task(ok_43())
        try:
            # Strict barrier: both users must be inside search_vacancies together.
            await asyncio.wait_for(
                asyncio.gather(entered_42.wait(), entered_43.wait()),
                timeout=5.0,
            )
            assert err_client.search_calls >= 1
            assert ok_client.search_calls >= 1
            assert not release.is_set()

            release.set()
            await asyncio.gather(task_42, task_43)
        finally:
            for t in (task_42, task_43):
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass

    asyncio.run(scenario())

    run_42 = database.latest_search_run(telegram_user_id=42)
    run_43 = database.latest_search_run(telegram_user_id=43)
    assert run_42 is not None
    assert run_43 is not None
    assert run_42.telegram_user_id == 42
    assert run_43.telegram_user_id == 43
    assert run_42.state == "paused_by_circuit_breaker"
    assert run_42.circuit_reason == "search_errors"
    assert run_43.state == "completed"
    assert run_43.circuit_reason == ""

    # No cross-user DB leakage.
    assert database.latest_search_run(telegram_user_id=42).state != "completed"
    assert database.latest_search_run(telegram_user_id=43).state == "completed"

    chat_ids = [chat_id for chat_id, _ in fake_bot.messages]
    texts_by_chat: dict[int, list[str]] = {}
    for chat_id, text in fake_bot.messages:
        texts_by_chat.setdefault(chat_id, []).append(text)

    # Circuit-breaker notification must be delivered with chat_id=42 only.
    assert 42 in chat_ids
    assert any(
        "приостановлен" in text.lower() or "search_errors" in text
        for text in texts_by_chat.get(42, [])
    )
    # User 43 must never receive user 42's circuit-breaker message.
    assert all(
        "search_errors" not in text and "приостановлен" not in text.lower()
        for text in texts_by_chat.get(43, [])
    )
    # Successful empty cycle for 43 produces its own completion notify with chat_id=43.
    assert 43 in chat_ids
    assert any(
        "цикл завершён" in text.lower() or "карточек" in text.lower()
        for text in texts_by_chat.get(43, [])
    )


# ============================================================================
# Phase 5.4 - HH account ownership verification (users A and B)
# ============================================================================


def test_phase54_ownership_isolation_with_separate_browser_profiles(
    tmp_path: Path,
) -> None:
    """User A cannot borrow user B's account, profile, or client identity."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    now = NOW

    _setup_user_with_profile_and_resume(database, 42, "Resume FortyTwo", now)
    _setup_user_with_profile_and_resume(database, 43, "Resume FortyThree", now)

    profile_42 = str(tmp_path / "browser-profiles" / "hh-account-42")
    profile_43 = str(tmp_path / "browser-profiles" / "hh-account-43")
    account_42_id = database.create_hh_account(
        telegram_user_id=42,
        account_label="Account 42",
        browser_profile_dir=profile_42,
        now=now,
    )
    account_43_id = database.create_hh_account(
        telegram_user_id=43,
        account_label="Account 43",
        browser_profile_dir=profile_43,
        now=now,
    )
    account_42 = database.get_hh_account_by_id(account_42_id, 42)
    account_43 = database.get_hh_account_by_id(account_43_id, 43)
    assert account_42 is not None
    assert account_43 is not None
    assert account_42.browser_profile_dir != account_43.browser_profile_dir

    # A cannot attach B's account to its runtime context.
    with pytest.raises(HHAccountOwnershipError, match="does not belong"):
        create_runtime_context(
            42, settings, database, hh_account=account_43
        )

    # A cannot borrow B's browser profile through a forged account copy.
    forged_profile = replace(
        account_42,
        browser_profile_dir=account_43.browser_profile_dir,
    )
    with pytest.raises(HHAccountOwnershipError, match="browser profile"):
        HHClient(
            FakeContext(ResumeTrackingPage()),
            settings,
            database,
            ApprovalGuard(settings, database),
            telegram_user_id=42,
            hh_account=forged_profile,
        )

    # A cannot construct an HHClient carrying B's account.
    with pytest.raises(HHAccountOwnershipError, match="does not belong"):
        HHClient(
            FakeContext(ResumeTrackingPage()),
            settings,
            database,
            ApprovalGuard(settings, database),
            telegram_user_id=42,
            hh_account=account_43,
        )

    # A's own client binds only A's account and profile.
    us_42 = load_user_settings(
        settings,
        app_mode="approval",
        enable_real_apply=True,
        max_applications_per_day=10,
        approval_ttl_minutes=30,
    )
    client_42 = HHClient(
        FakeContext(ResumeTrackingPage()),
        settings,
        database,
        ApprovalGuard(us_42, database, now_factory=lambda: now + timedelta(minutes=2)),
        telegram_user_id=42,
        user_settings=us_42,
    )
    assert client_42.hh_account is not None
    assert client_42.hh_account.id == account_42.id
    assert client_42.hh_account.telegram_user_id == 42
    assert client_42.hh_account.browser_profile_dir == profile_42
    assert client_42.hh_account.browser_profile_dir != profile_43

    # B's normal workflow is unaffected.
    permit_43 = _setup_vacancy_and_approve(database, "job-43", 43, now)
    page_43 = ResumeTrackingPage(expected_resume="Resume FortyThree")
    us_43 = load_user_settings(
        settings,
        app_mode="approval",
        enable_real_apply=True,
        max_applications_per_day=10,
        approval_ttl_minutes=30,
    )
    client_43 = HHClient(
        FakeContext(page_43),
        settings,
        database,
        ApprovalGuard(us_43, database, now_factory=lambda: now + timedelta(minutes=2)),
        telegram_user_id=43,
        user_settings=us_43,
        hh_account=account_43,
        sleep=lambda _: asyncio.sleep(0),
        now_factory=lambda: now + timedelta(minutes=3),
    )
    assert client_43.hh_account is not None
    assert client_43.hh_account.id == account_43.id

    result_43 = asyncio.run(
        client_43.submit_application(
            ApplicationPermission("job-43", permit_43, 43)
        )
    )

    assert result_43 is True
    assert "selected_resume:Resume FortyThree" in page_43.actions
    assert database.get("job-43").status is VacancyStatus.APPLIED
