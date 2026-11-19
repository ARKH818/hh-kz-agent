import dataclasses
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from database import (
    HHAccount,
    HHAccountStatus,
    HH_ACCOUNT_TRANSITIONS,
    Database,
    VacancyStatus,
    is_hh_account_authorized,
    normalize_hh_account_status,
    validate_hh_account_transition,
)


NOW = datetime(2026, 7, 26, 9, 0, tzinfo=UTC)


def make_database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "agent.db")
    database.init()
    return database


def discover(database: Database, job_id: str = "job-1") -> None:
    assert database.discover(
        job_id=job_id,
        title="Python developer",
        company="Example",
        url=f"https://example.com/vacancy/{job_id}",
        description_hash="abc123",
        search_query="Python",
        discovered_at=NOW,
    )


def request_approval(database: Database, job_id: str = "job-1") -> None:
    assert database.request_approval(
        job_id=job_id,
        now=NOW,
    )


def approve(database: Database, job_id: str = "job-1") -> str:
    discover(database, job_id)
    request_approval(database, job_id)

    token = database.approve(
        job_id=job_id,
        telegram_user_id=42,
        expected_user_id=42,
        now=NOW + timedelta(minutes=1),
    )
    assert token
    return token


def claim(database: Database, job_id: str, token: str, **overrides: object):
    arguments = {
        "job_id": job_id,
        "permit": token,
        "telegram_user_id": 42,
        "expected_user_id": 42,
        "app_mode": "approval",
        "enable_real_apply": True,
        "daily_limit": 5,
        "now": NOW + timedelta(minutes=2),
    }
    arguments.update(overrides)
    return database.claim_application(**arguments)


def test_status_transitions_are_conditional_and_terminal(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    discover(database)

    assert database.get("job-1").status is VacancyStatus.DISCOVERED
    assert database.transition(
        "job-1", VacancyStatus.DISCOVERED, VacancyStatus.REJECTED_BY_FILTER
    )
    assert not database.transition(
        "job-1", VacancyStatus.DISCOVERED, VacancyStatus.PENDING_APPROVAL
    )
    assert database.get("job-1").status is VacancyStatus.REJECTED_BY_FILTER


def test_duplicate_discovery_is_rejected(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    discover(database)

    assert not database.discover(
        job_id="job-1",
        title="Duplicate",
        company="Other",
        url="https://example.com/duplicate",
        description_hash="different",
        search_query="Other",
        discovered_at=NOW,
    )
    assert database.get("job-1").title == "Python developer"


def test_unprocessed_discovered_returns_discovered_rows(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    discover(database, "job-1")
    discover(database, "job-2")
    assert database.transition(
        "job-2", VacancyStatus.DISCOVERED, VacancyStatus.REJECTED_BY_FILTER
    )

    assert [row.id for row in database.unprocessed_discovered()] == ["job-1"]


def test_company_details_are_persisted(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    assert database.discover(
        job_id="job-1",
        title="Python developer",
        company="Example",
        company_url="https://hh.ru/employer/123",
        url="https://hh.ru/vacancy/1",
        description_hash="abc123",
        search_query="Python",
        discovered_at=NOW,
    )
    assert database.store_company_details("job-1", rating=4.7, reviews_count=128)

    vacancy = database.get("job-1")
    assert vacancy.company_url == "https://hh.ru/employer/123"
    assert (vacancy.company_rating, vacancy.company_reviews_count) == (4.7, 128)


def test_search_run_survives_reopen_with_aggregated_reasons(tmp_path: Path) -> None:
    path = tmp_path / "agent.db"
    database = Database(path)
    database.init()

    database.save_search_run(
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=12),
        state="completed",
        query_count=2,
        found_results=9,
        new_vacancies=4,
        duplicates=5,
        rejected_by_filter=2,
        telegram_cards=0,
        error_count=1,
        rejection_reasons={"title": 2},
        error_reasons={"network_error": 1},
        last_safe_error="network_error",
    )

    run = Database(path).latest_search_run()

    assert run is not None
    assert run.state == "completed"
    assert run.found_results == 9
    assert run.rejection_reasons == {"title": 2}
    assert run.error_reasons == {"network_error": 1}


def test_search_run_can_record_notification_error(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    run_id = database.save_search_run(
        started_at=NOW,
        finished_at=NOW,
        state="completed",
        query_count=1,
        found_results=0,
        new_vacancies=0,
        duplicates=0,
        rejected_by_filter=0,
        telegram_cards=0,
        error_count=0,
        rejection_reasons={},
        error_reasons={},
    )

    assert database.record_search_run_error(run_id, "telegram_error")
    run = database.latest_search_run()
    assert run is not None
    assert run.error_count == 1
    assert run.error_reasons == {"telegram_error": 1}
    assert run.last_safe_error == "telegram_error"


def test_pending_card_does_not_expire_before_user_click(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    discover(database)
    request_approval(database)

    assert database.get("job-1").approval_expires_at is None
    assert database.expire_approved(NOW + timedelta(days=1)) == 0
    assert database.get("job-1").status is VacancyStatus.PENDING_APPROVAL


def test_approval_click_starts_permission_ttl(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    discover(database)
    request_approval(database)

    token = database.approve(
        "job-1", 42, 42, NOW + timedelta(hours=8), ttl_minutes=30
    )

    assert token
    vacancy = database.get("job-1")
    assert vacancy.status is VacancyStatus.APPROVED
    assert vacancy.approval_expires_at == (
        NOW + timedelta(hours=8, minutes=30)
    ).isoformat()


def test_foreign_user_cannot_approve_or_skip(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    discover(database)
    request_approval(database)

    assert (
        database.approve(
            job_id="job-1",
            telegram_user_id=99,
            expected_user_id=42,
            now=NOW + timedelta(minutes=1),
        )
        is None
    )
    assert not database.skip("job-1", telegram_user_id=99, expected_user_id=42)
    assert database.get("job-1").status is VacancyStatus.PENDING_APPROVAL


def test_claim_rechecks_mode_feature_flag_and_user(tmp_path: Path) -> None:
    for name, override in (
        ("dry", {"app_mode": "dry_run"}),
        ("disabled", {"enable_real_apply": False}),
        ("foreign", {"telegram_user_id": 99}),
    ):
        database = Database(tmp_path / f"{name}.db")
        database.init()
        token = approve(database)

        result = claim(database, "job-1", token, **override)

        assert not result.allowed
        assert database.get("job-1").status is VacancyStatus.APPROVED


def test_invalid_or_reused_permit_cannot_claim_twice(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    token = approve(database)

    assert not claim(database, "job-1", "forged").allowed
    assert claim(database, "job-1", token).allowed
    assert not claim(database, "job-1", token).allowed
    assert database.get("job-1").status is VacancyStatus.APPLYING


def test_expired_permission_is_blocked(tmp_path: Path) -> None:
    expired_database = Database(tmp_path / "expired.db")
    expired_database.init()
    token = approve(expired_database)
    result = claim(
        expired_database,
        "job-1",
        token,
        now=NOW + timedelta(minutes=31),
    )
    assert not result.allowed
    assert expired_database.get("job-1").status is VacancyStatus.EXPIRED


def test_daily_limit_uses_persisted_applied_rows(tmp_path: Path) -> None:
    path = tmp_path / "agent.db"
    database = Database(path)
    database.init()
    first_token = approve(database, "job-1")
    assert claim(database, "job-1", first_token, daily_limit=1).allowed
    assert database.complete_application(
        "job-1", first_token, success=True, now=NOW + timedelta(minutes=3)
    )

    reopened = Database(path)
    second_token = approve(reopened, "job-2")
    second_claim = claim(reopened, "job-2", second_token, daily_limit=1)

    assert reopened.applied_today(NOW + timedelta(minutes=4)) == 1
    assert not second_claim.allowed
    assert reopened.get("job-2").status is VacancyStatus.APPROVED


def test_daily_limit_atomically_counts_in_progress_claims(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    first_token = approve(database, "job-1")
    second_token = approve(database, "job-2")

    assert claim(database, "job-1", first_token, daily_limit=1).allowed
    second_claim = claim(database, "job-2", second_token, daily_limit=1)

    assert not second_claim.allowed
    assert second_claim.reason == "daily_limit"
    assert database.get("job-2").status is VacancyStatus.APPROVED


def test_only_the_claimant_permit_can_complete_application(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    token = approve(database)
    assert claim(database, "job-1", token).allowed

    assert not database.complete_application(
        "job-1", "forged", success=True, now=NOW + timedelta(minutes=3)
    )
    assert database.get("job-1").status is VacancyStatus.APPLYING
    assert database.complete_application(
        "job-1", token, success=True, now=NOW + timedelta(minutes=3)
    )


def test_failed_final_submit_attempt_keeps_daily_reservation(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    first_token = approve(database, "job-1")
    second_token = approve(database, "job-2")
    assert claim(database, "job-1", first_token, daily_limit=1).allowed
    assert database.mark_submit_attempt(
        "job-1", first_token, now=NOW + timedelta(minutes=3), daily_limit=1
    )
    assert database.complete_application(
        "job-1",
        first_token,
        success=False,
        now=NOW + timedelta(minutes=3),
        error_text="success signal missing",
    )

    assert not claim(database, "job-2", second_token, daily_limit=1).allowed


def test_concurrent_claim_has_one_winner(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    token = approve(database)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(database, "job-1", token), range(2)))

    assert [result.allowed for result in results].count(True) == 1
    assert database.get("job-1").status is VacancyStatus.APPLYING


def test_legacy_applied_jobs_is_preserved_and_not_imported(tmp_path: Path) -> None:
    path = tmp_path / "agent.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE applied_jobs (id TEXT PRIMARY KEY, title TEXT, url TEXT, applied_at TEXT)"
        )
        connection.execute(
            "INSERT INTO applied_jobs VALUES (?, ?, ?, ?)",
            ("legacy", "Rejected in old version", "https://example.com/legacy", "2026-07-20"),
        )

    database = Database(path)
    database.init()

    assert database.get("legacy") is None
    assert database.applied_today(NOW) == 0
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT title FROM applied_jobs").fetchone()[0] == "Rejected in old version"


def test_legacy_vacancy_survives_status_migration(tmp_path: Path) -> None:
    path = tmp_path / "agent.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            f"""
            CREATE TABLE vacancies (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                company TEXT NOT NULL DEFAULT '',
                url TEXT NOT NULL,
                description_hash TEXT NOT NULL DEFAULT '',
                search_query TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL CHECK (status IN ('discovered', 'pending_approval')),
                discovered_at TEXT NOT NULL,
                approval_requested_at TEXT,
                approval_expires_at TEXT,
                approved_at TEXT,
                applied_at TEXT,
                approver_id INTEGER,
                permit_hash TEXT,
                error_text TEXT NOT NULL DEFAULT ''
            )
            """
        )
        connection.execute(
            """
            INSERT INTO vacancies (
                id, title, company, url, description_hash, search_query,
                status, discovered_at, approval_requested_at,
                approval_expires_at, approved_at, applied_at, approver_id,
                permit_hash, error_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-job",
                "Legacy developer",
                "Legacy Co",
                "https://example.com/legacy",
                "legacy-hash",
                "Python",
                VacancyStatus.DISCOVERED.value,
                "2026-07-20T09:00:00+00:00",
                "requested",
                "expires",
                "approved",
                "applied",
                42,
                "permit",
                "legacy error",
            ),
        )

    database = Database(path)
    database.init()

    vacancy = database.get("legacy-job")
    assert vacancy.title == "Legacy developer"
    assert vacancy.status is VacancyStatus.DISCOVERED
    assert vacancy.company_url == ""
    assert vacancy.approval_requested_at == "requested"
    assert vacancy.approval_expires_at == "expires"
    assert vacancy.approved_at == "approved"
    assert vacancy.applied_at == "applied"
    assert vacancy.approver_id == 42
    assert vacancy.error_text == "legacy error"


def test_get_or_create_user_creates_new_user(tmp_path: Path) -> None:
    database = make_database(tmp_path)

    user = database.get_or_create_user(
        telegram_user_id=42,
        username="arman",
        first_name="Арман",
        now=NOW,
    )

    assert user["telegram_user_id"] == 42
    assert user["username"] == "arman"
    assert user["first_name"] == "Арман"
    assert user["is_active"] == 1


def test_get_or_create_user_does_not_duplicate_existing_user(
    tmp_path: Path,
) -> None:
    database = make_database(tmp_path)

    first = database.get_or_create_user(
        telegram_user_id=42,
        username="arman",
        first_name="Арман",
        now=NOW,
    )
    second = database.get_or_create_user(
        telegram_user_id=42,
        username="arman",
        first_name="Арман",
        now=NOW + timedelta(minutes=1),
    )

    assert first["telegram_user_id"] == second["telegram_user_id"]

    with sqlite3.connect(database.path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM users WHERE telegram_user_id = ?",
            (42,),
        ).fetchone()[0]

    assert count == 1


def test_get_or_create_user_updates_existing_user(
    tmp_path: Path,
) -> None:
    database = make_database(tmp_path)

    database.create_user(
        telegram_user_id=42,
        username="old_name",
        first_name="Old",
        now=NOW,
    )

    user = database.get_or_create_user(
        telegram_user_id=42,
        username="new_name",
        first_name="New",
        now=NOW + timedelta(minutes=1),
    )

    assert user["username"] == "new_name"
    assert user["first_name"] == "New"


def test_users_are_isolated(tmp_path: Path) -> None:
    database = make_database(tmp_path)

    database.get_or_create_user(
        telegram_user_id=42,
        username="user42",
        first_name="User 42",
        now=NOW,
    )
    database.get_or_create_user(
        telegram_user_id=43,
        username="user43",
        first_name="User 43",
        now=NOW,
    )

    user42 = database.get_user(42)
    user43 = database.get_user(43)

    assert user42 is not None
    assert user43 is not None
    assert user42["telegram_user_id"] == 42
    assert user43["telegram_user_id"] == 43
    assert user42["username"] != user43["username"]


# =============================================================================
# Cross-user isolation tests (Stage 1)
# =============================================================================

# -- A. User A cannot get data of User B --

def test_user_a_cannot_get_user_b_profile(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, username="user_a", now=NOW)
    database.create_user(telegram_user_id=99, username="user_b", now=NOW)

    database.create_user_profile(
        telegram_user_id=42, name="Alice", location="Moscow", now=NOW,
    )
    database.create_user_profile(
        telegram_user_id=99, name="Bob", location="SPb", now=NOW,
    )

    # get_user_profile is scoped by user_id - returns correct data for each user
    profile_a = database.get_user_profile(42)
    profile_b = database.get_user_profile(99)
    assert profile_a["name"] == "Alice"
    assert profile_b["name"] == "Bob"


def test_user_a_cannot_get_user_b_search_profile(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    profile_id = database.create_search_profile(
        telegram_user_id=42, name="Python jobs", now=NOW,
    )

    # User B cannot access User A profile (wrong user_id in WHERE)
    assert database.get_search_profile(profile_id, telegram_user_id=99) is None


def test_user_a_cannot_get_user_b_resume(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    resume_id = database.create_resume(
        telegram_user_id=42, hh_resume_id="resume-123", now=NOW,
    )

    assert database.get_resume(resume_id, telegram_user_id=99) is None


def test_user_a_cannot_get_user_b_application(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    database.discover(
        job_id="job-1", title="Dev", company="Co",
        url="https://example.com/job-1", description_hash="hash",
        search_query="Python", discovered_at=NOW,
    )

    app_id = database.create_application(
        telegram_user_id=42, vacancy_id="job-1", now=NOW,
    )
    assert app_id > 0

    assert database.get_application(app_id, telegram_user_id=99) is None


def test_user_a_cannot_get_user_b_scheduled_task(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    task_id = database.create_scheduled_task(
        telegram_user_id=42, task_type="search", now=NOW,
    )

    assert database.get_scheduled_task(task_id, telegram_user_id=99) is None


# -- B. User A cannot modify data of User B --

def test_user_a_cannot_modify_user_b_search_profile(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    profile_id = database.create_search_profile(
        telegram_user_id=42, name="Python jobs", now=NOW,
    )

    # User B cannot modify User A profile (wrong user_id in WHERE)
    assert not database.update_search_profile(
        profile_id, telegram_user_id=99, name="Hacked"
    )
    # Verify User A profile intact
    profile = database.get_search_profile(profile_id, telegram_user_id=42)
    assert profile["name"] == "Python jobs"


def test_user_a_cannot_delete_user_b_search_profile(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    profile_id = database.create_search_profile(
        telegram_user_id=42, name="Python jobs", now=NOW,
    )

    assert not database.delete_search_profile(profile_id, telegram_user_id=99)
    assert database.get_search_profile(profile_id, telegram_user_id=42) is not None


def test_user_a_cannot_modify_user_b_resume(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    resume_id = database.create_resume(
        telegram_user_id=42, hh_resume_id="resume-123", title="Dev", now=NOW,
    )

    assert not database.update_resume(resume_id, telegram_user_id=99, title="Hacked")
    assert database.get_resume(resume_id, telegram_user_id=42)["title"] == "Dev"


def test_user_a_cannot_delete_user_b_resume(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    resume_id = database.create_resume(
        telegram_user_id=42, hh_resume_id="resume-123", now=NOW,
    )

    assert not database.delete_resume(resume_id, telegram_user_id=99)
    assert database.get_resume(resume_id, telegram_user_id=42) is not None


def test_user_a_cannot_modify_user_b_application(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    database.discover(
        job_id="job-1", title="Dev", company="Co",
        url="https://example.com/job-1", description_hash="hash",
        search_query="Python", discovered_at=NOW,
    )

    app_id = database.create_application(
        telegram_user_id=42, vacancy_id="job-1", now=NOW,
    )

    assert not database.update_application(app_id, telegram_user_id=99, status="applied")
    assert database.get_application(app_id, telegram_user_id=42)["status"] == "new"


def test_user_a_cannot_modify_user_b_scheduled_task(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    task_id = database.create_scheduled_task(
        telegram_user_id=42, task_type="search", now=NOW,
    )

    assert not database.update_scheduled_task(task_id, telegram_user_id=99, status="running")
    assert database.get_scheduled_task(task_id, telegram_user_id=42)["status"] == "pending"


def test_user_a_cannot_delete_user_b_scheduled_task(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    task_id = database.create_scheduled_task(
        telegram_user_id=42, task_type="search", now=NOW,
    )

    assert not database.delete_scheduled_task(task_id, telegram_user_id=99)
    assert database.get_scheduled_task(task_id, telegram_user_id=42) is not None


# -- C. Application events are user-scoped --

def test_application_events_are_user_scoped(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    database.discover(
        job_id="job-1", title="Dev", company="Co",
        url="https://example.com/job-1", description_hash="hash",
        search_query="Python", discovered_at=NOW,
    )

    app_id = database.create_application(
        telegram_user_id=42, vacancy_id="job-1", now=NOW,
    )

    # User A can add and read events
    event_id = database.add_application_event(app_id, 42, "created", "Application created")
    assert event_id > 0
    events = database.get_application_events(app_id, 42)
    assert len(events) == 1

    # User B cannot add events to User A application
    assert database.add_application_event(app_id, 99, "hacked", "Hacked") == 0

    # User B cannot read User A events
    assert database.get_application_events(app_id, 99) == []


# -- D. HH account is per-user --

def test_user_a_cannot_get_user_b_hh_account(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    database.create_hh_account(telegram_user_id=42, account_label="main", now=NOW)

    assert database.get_hh_account(99) is None
    account = database.get_hh_account(42)
    assert account is not None
    assert account["account_label"] == "main"


def test_user_a_cannot_modify_user_b_hh_account(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    database.create_hh_account(telegram_user_id=42, account_label="main", now=NOW)

    # User B cannot modify User A account (wrong user_id in WHERE)
    assert not database.update_hh_account(99, account_label="hacked")
    assert database.get_hh_account(42)["account_label"] == "main"


# -- E. Resume is per-user --

def test_resume_ownership_is_enforced(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    resume_id = database.create_resume(
        telegram_user_id=42, hh_resume_id="resume-123", title="Dev", now=NOW,
    )

    # User B cannot access
    assert database.get_resume(resume_id, 99) is None
    assert not database.update_resume(resume_id, 99, title="Hacked")
    assert not database.delete_resume(resume_id, 99)

    # User A can access
    assert database.get_resume(resume_id, 42) is not None


# -- F. Simultaneous work of two users does not mix data --

def test_two_users_data_do_not_mix(tmp_path: Path) -> None:
    database = make_database(tmp_path)

    for uid, name in ((42, "Alice"), (99, "Bob")):
        database.create_user(telegram_user_id=uid, username=name, now=NOW)
        database.create_user_profile(telegram_user_id=uid, name=name, now=NOW)
        database.create_search_profile(telegram_user_id=uid, name=f"{name} jobs", now=NOW)
        database.create_hh_account(telegram_user_id=uid, account_label=name, now=NOW)
        database.create_resume(telegram_user_id=uid, hh_resume_id=f"resume-{uid}", now=NOW)
        database.create_scheduled_task(telegram_user_id=uid, task_type="search", now=NOW)

    # Verify isolation: each user sees only their own data
    for uid, name in ((42, "Alice"), (99, "Bob")):
        assert database.get_user_profile(uid)["name"] == name
        assert len(database.get_search_profiles(uid)) == 1
        assert database.get_hh_account(uid)["account_label"] == name
        assert len(database.get_resumes(uid)) == 1
        assert len(database.get_scheduled_tasks(uid)) == 1

    # Verify no cross-contamination in counts
    with sqlite3.connect(database.path) as connection:
        for table in ("users", "user_profiles", "search_profiles", "hh_accounts", "resumes", "scheduled_tasks"):
            count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count == 2, f"{table} has {count} rows, expected 2"


# -- G. Re-init preserves existing data --

def test_reinit_preserves_existing_user_data(tmp_path: Path) -> None:
    path = tmp_path / "agent.db"
    database = Database(path)
    database.init()

    database.create_user(telegram_user_id=42, username="alice", now=NOW)
    database.create_user_profile(telegram_user_id=42, name="Alice", now=NOW)
    profile_id = database.create_search_profile(
        telegram_user_id=42, name="Python", search_queries=["Python"], now=NOW
    )
    database.create_hh_account(telegram_user_id=42, account_label="main", now=NOW)
    resume_id = database.create_resume(
        telegram_user_id=42, hh_resume_id="r1", title="Dev", now=NOW
    )
    task_id = database.create_scheduled_task(
        telegram_user_id=42, task_type="search", now=NOW
    )

    # Re-init the database
    database2 = Database(path)
    database2.init()

    # All data preserved
    assert database2.get_user(42) is not None
    assert database2.get_user_profile(42) is not None
    assert database2.get_search_profile(profile_id, 42) is not None
    assert database2.get_hh_account(42) is not None
    assert database2.get_resume(resume_id, 42) is not None
    assert database2.get_scheduled_task(task_id, 42) is not None


# -- List operations return only owned data --

def test_search_profile_list_returns_only_owned_profiles(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    database.create_search_profile(telegram_user_id=42, name="A", now=NOW)
    database.create_search_profile(telegram_user_id=42, name="B", now=NOW)
    database.create_search_profile(telegram_user_id=99, name="C", now=NOW)

    profiles_42 = database.get_search_profiles(42)
    profiles_99 = database.get_search_profiles(99)

    assert len(profiles_42) == 2
    assert len(profiles_99) == 1
    assert {p["name"] for p in profiles_42} == {"A", "B"}
    assert profiles_99[0]["name"] == "C"


def test_resume_list_returns_only_owned_resumes(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    database.create_resume(telegram_user_id=42, hh_resume_id="r1", now=NOW)
    database.create_resume(telegram_user_id=42, hh_resume_id="r2", now=NOW)
    database.create_resume(telegram_user_id=99, hh_resume_id="r3", now=NOW)

    resumes_42 = database.get_resumes(42)
    resumes_99 = database.get_resumes(99)

    assert len(resumes_42) == 2
    assert len(resumes_99) == 1


def test_application_list_returns_only_owned_applications(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    for job_id in ("job-1", "job-2", "job-3"):
        database.discover(
            job_id=job_id, title="Dev", company="Co",
            url=f"https://example.com/{job_id}", description_hash="hash",
            search_query="Python", discovered_at=NOW,
        )

    database.create_application(telegram_user_id=42, vacancy_id="job-1", now=NOW)
    database.create_application(telegram_user_id=42, vacancy_id="job-2", now=NOW)
    database.create_application(telegram_user_id=99, vacancy_id="job-3", now=NOW)

    apps_42 = database.get_applications(42)
    apps_99 = database.get_applications(99)

    assert len(apps_42) == 2
    assert len(apps_99) == 1


def test_scheduled_task_list_returns_only_owned_tasks(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    database.create_scheduled_task(telegram_user_id=42, task_type="search", now=NOW)
    database.create_scheduled_task(telegram_user_id=42, task_type="raise", now=NOW)
    database.create_scheduled_task(telegram_user_id=99, task_type="search", now=NOW)

    tasks_42 = database.get_scheduled_tasks(42)
    tasks_99 = database.get_scheduled_tasks(99)

    assert len(tasks_42) == 2
    assert len(tasks_99) == 1


def test_user_profile_update_is_scoped(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    database.create_user_profile(telegram_user_id=42, name="Alice", location="Moscow", now=NOW)
    database.create_user_profile(telegram_user_id=99, name="Bob", location="SPb", now=NOW)

    # User 42 updates only their profile
    assert database.update_user_profile(42, name="Alice Updated")
    assert database.get_user_profile(42)["name"] == "Alice Updated"
    assert database.get_user_profile(99)["name"] == "Bob"  # Unchanged


def test_hh_account_upsert_behavior(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)

    # First creation
    account_id_1 = database.create_hh_account(
        telegram_user_id=42, account_label="main", now=NOW
    )

    # Second creation for same user should update (upsert)
    account_id_2 = database.create_hh_account(
        telegram_user_id=42, account_label="updated", now=NOW
    )

    account = database.get_hh_account(42)
    assert account is not None
    assert account["account_label"] == "updated"
    # Same row updated, not new row
    assert account_id_1 == account_id_2


# =============================================================================
# Stage 3.2: search_runs user-scoped tests
# =============================================================================

def test_latest_search_run_is_user_scoped(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=43, now=NOW)

    database.save_search_run(
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=10),
        state="completed",
        query_count=1,
        found_results=5,
        new_vacancies=2,
        duplicates=3,
        rejected_by_filter=0,
        telegram_cards=2,
        error_count=0,
        rejection_reasons={},
        error_reasons={},
        telegram_user_id=42,
    )
    database.save_search_run(
        started_at=NOW + timedelta(minutes=1),
        finished_at=NOW + timedelta(minutes=1, seconds=10),
        state="completed",
        query_count=2,
        found_results=10,
        new_vacancies=5,
        duplicates=5,
        rejected_by_filter=1,
        telegram_cards=3,
        error_count=0,
        rejection_reasons={},
        error_reasons={},
        telegram_user_id=43,
    )

    run_42 = database.latest_search_run(telegram_user_id=42)
    run_43 = database.latest_search_run(telegram_user_id=43)

    assert run_42 is not None
    assert run_43 is not None
    assert run_42.telegram_user_id == 42
    assert run_43.telegram_user_id == 43
    assert run_42.found_results == 5
    assert run_43.found_results == 10


def test_save_search_run_stores_user_id(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)

    run_id = database.save_search_run(
        started_at=NOW,
        finished_at=NOW,
        state="completed",
        query_count=1,
        found_results=0,
        new_vacancies=0,
        duplicates=0,
        rejected_by_filter=0,
        telegram_cards=0,
        error_count=0,
        rejection_reasons={},
        error_reasons={},
        telegram_user_id=42,
    )

    run = database.latest_search_run(telegram_user_id=42)
    assert run is not None
    assert run.id == run_id
    assert run.telegram_user_id == 42


def test_record_search_run_error_is_user_scoped(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=43, now=NOW)

    run_id_42 = database.save_search_run(
        started_at=NOW,
        finished_at=NOW,
        state="completed",
        query_count=1,
        found_results=0,
        new_vacancies=0,
        duplicates=0,
        rejected_by_filter=0,
        telegram_cards=0,
        error_count=0,
        rejection_reasons={},
        error_reasons={},
        telegram_user_id=42,
    )
    run_id_43 = database.save_search_run(
        started_at=NOW,
        finished_at=NOW,
        state="completed",
        query_count=1,
        found_results=0,
        new_vacancies=0,
        duplicates=0,
        rejected_by_filter=0,
        telegram_cards=0,
        error_count=0,
        rejection_reasons={},
        error_reasons={},
        telegram_user_id=43,
    )

    assert database.record_search_run_error(run_id_42, "network_error", telegram_user_id=42)
    assert not database.record_search_run_error(run_id_42, "hacked", telegram_user_id=43)

    run_42 = database.latest_search_run(telegram_user_id=42)
    run_43 = database.latest_search_run(telegram_user_id=43)

    assert run_42.error_count == 1
    assert run_42.last_safe_error == "network_error"
    assert run_43.error_count == 0
    assert run_43.last_safe_error == ""


def test_search_run_backward_compatibility_without_user_id(tmp_path: Path) -> None:
    database = make_database(tmp_path)

    run_id = database.save_search_run(
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=12),
        state="completed",
        query_count=2,
        found_results=9,
        new_vacancies=4,
        duplicates=5,
        rejected_by_filter=2,
        telegram_cards=0,
        error_count=1,
        rejection_reasons={"title": 2},
        error_reasons={"network_error": 1},
        last_safe_error="network_error",
    )

    run = database.latest_search_run()
    assert run is not None
    assert run.id == run_id
    assert run.telegram_user_id is None
    assert run.state == "completed"
    assert run.found_results == 9

    assert database.record_search_run_error(run_id, "telegram_error")
    run = database.latest_search_run()
    assert run.error_count == 2


# =============================================================================
# Stage 3.3: get_active_profile tests
# =============================================================================

def test_get_active_profile_full_assembly(tmp_path: Path) -> None:
    """A. Полная сборка профиля одного пользователя."""
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, username="testuser", now=NOW)
    database.create_user_profile(
        telegram_user_id=42,
        name="Арман",
        location="Алматы",
        experience_summary="Опыт в логистике",
        education="Бакалавр",
        github_url="https://github.com/arman",
        salary_expectation="300000 KZT",
        additional_information="Полная занятость",
        now=NOW,
    )
    database.create_search_profile(
        telegram_user_id=42,
        name="Логистика",
        search_queries=["Менеджер по логистике", "Менеджер по ВЭД"],
        areas=["160"],
        experience_filters=["between1And3"],
        desired_positions=["Менеджер по логистике"],
        technologies=["Excel", "1С"],
        excluded_positions=["Курьер"],
        work_format=["На месте работодателя"],
        enabled=True,
        now=NOW,
    )
    database.create_resume(
        telegram_user_id=42,
        hh_resume_id="resume-123",
        title="Менеджер по логистике",
        is_active=True,
        now=NOW,
    )

    profile = database.get_active_profile(42)

    assert profile is not None
    # Candidate fields
    assert profile.candidate.name == "Арман"
    assert profile.candidate.location == "Алматы"
    assert profile.candidate.desired_positions == ("Менеджер по логистике",)
    assert profile.candidate.experience_summary == "Опыт в логистике"
    assert profile.candidate.education == "Бакалавр"
    assert profile.candidate.technologies == ("Excel", "1С")
    assert profile.candidate.projects == ()
    assert profile.candidate.github_url == "https://github.com/arman"
    assert profile.candidate.salary_expectation == "300000 KZT"
    assert profile.candidate.work_format == ("На месте работодателя",)
    assert profile.candidate.excluded_positions == ("Курьер",)
    assert profile.candidate.additional_information == "Полная занятость"
    # HH fields
    assert profile.hh.resume_name == "Менеджер по логистике"
    assert profile.hh.search_queries == ("Менеджер по логистике", "Менеджер по ВЭД")
    assert profile.hh.areas == ("160",)
    assert profile.hh.experience_filters == ("between1And3",)


def test_get_active_profile_user_isolation(tmp_path: Path) -> None:
    """B. Изоляция пользователей."""
    database = make_database(tmp_path)

    # User 42
    database.create_user(telegram_user_id=42, username="user42", now=NOW)
    database.create_user_profile(
        telegram_user_id=42, name="User 42", location="Almaty", now=NOW,
    )
    database.create_search_profile(
        telegram_user_id=42, name="Profile 42",
        search_queries=["Query 42"], areas=["160"],
        desired_positions=["Position 42"], technologies=["Tech 42"],
        work_format=["Remote"], excluded_positions=["Exclude 42"],
        experience_filters=["noExperience"],
        enabled=True, now=NOW,
    )
    database.create_resume(
        telegram_user_id=42, hh_resume_id="r42", title="Resume 42",
        is_active=True, now=NOW,
    )

    # User 43
    database.create_user(telegram_user_id=43, username="user43", now=NOW)
    database.create_user_profile(
        telegram_user_id=43, name="User 43", location="Astana", now=NOW,
    )
    database.create_search_profile(
        telegram_user_id=43, name="Profile 43",
        search_queries=["Query 43"], areas=["1"],
        desired_positions=["Position 43"], technologies=["Tech 43"],
        work_format=["Office"], excluded_positions=["Exclude 43"],
        experience_filters=["between3And6"],
        enabled=True, now=NOW,
    )
    database.create_resume(
        telegram_user_id=43, hh_resume_id="r43", title="Resume 43",
        is_active=True, now=NOW,
    )

    profile = database.get_active_profile(42)

    assert profile is not None
    assert profile.candidate.name == "User 42"
    assert profile.candidate.location == "Almaty"
    assert profile.hh.resume_name == "Resume 42"
    assert profile.hh.search_queries == ("Query 42",)
    assert profile.hh.areas == ("160",)
    # Убеждаемся, что данные 43 не просочились
    assert profile.candidate.name != "User 43"
    assert profile.hh.resume_name != "Resume 43"


def test_get_active_profile_missing_user(tmp_path: Path) -> None:
    """C. Отсутствие пользователя."""
    database = make_database(tmp_path)

    result = database.get_active_profile(99999)

    assert result is None


def test_get_active_profile_selects_enabled_search_profile(tmp_path: Path) -> None:
    """D. Несколько search_profiles -- выбирается активный (enabled=1)."""
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user_profile(telegram_user_id=42, name="Test", now=NOW)

    # Первый профиль -- disabled
    database.create_search_profile(
        telegram_user_id=42, name="Disabled",
        search_queries=["Disabled Query"], areas=["1"],
        desired_positions=["Disabled Pos"], enabled=False, now=NOW,
    )
    # Второй профиль -- enabled (должен быть выбран)
    database.create_search_profile(
        telegram_user_id=42, name="Enabled",
        search_queries=["Enabled Query"], areas=["160"],
        desired_positions=["Enabled Pos"], enabled=True, now=NOW,
    )
    # Третий профиль -- тоже enabled, но создан позже (по id ASC первый enabled -- второй)
    database.create_search_profile(
        telegram_user_id=42, name="Also Enabled",
        search_queries=["Also Enabled"], areas=["2"],
        desired_positions=["Also Pos"], enabled=True, now=NOW,
    )

    database.create_resume(
        telegram_user_id=42, hh_resume_id="r1", title="Resume", is_active=True, now=NOW,
    )

    profile = database.get_active_profile(42)

    assert profile is not None
    assert profile.hh.search_queries == ("Enabled Query",)
    assert profile.hh.areas == ("160",)
    assert profile.candidate.desired_positions == ("Enabled Pos",)


def test_get_active_profile_selects_active_resume(tmp_path: Path) -> None:
    """E. Несколько resumes -- выбирается активное (is_active=1)."""
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user_profile(telegram_user_id=42, name="Test", now=NOW)
    database.create_search_profile(
        telegram_user_id=42, name="Profile", enabled=True, now=NOW,
    )

    # Первое резюме -- inactive
    database.create_resume(
        telegram_user_id=42, hh_resume_id="r1", title="Inactive Resume",
        is_active=False, now=NOW,
    )
    # Второе резюме -- active (должно быть выбрано)
    database.create_resume(
        telegram_user_id=42, hh_resume_id="r2", title="Active Resume",
        is_active=True, now=NOW,
    )
    # Третье резюме -- тоже active, но позже (по id ASC первое active -- второе)
    database.create_resume(
        telegram_user_id=42, hh_resume_id="r3", title="Also Active",
        is_active=True, now=NOW,
    )

    profile = database.get_active_profile(42)

    assert profile is not None
    assert profile.hh.resume_name == "Active Resume"


# =============================================================================
# Phase 5.1: HHAccount domain model tests
# =============================================================================


def _create_hh_account(database: Database) -> None:
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_hh_account(
        telegram_user_id=42,
        account_label="main",
        browser_profile_dir="/tmp/profile",
        now=NOW,
    )


def test_hh_account_from_db_row(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    _create_hh_account(database)

    row = database.get_hh_account(42)
    assert row is not None

    account = Database._hh_account(row)

    assert account is not None
    assert isinstance(account, HHAccount)
    assert account.telegram_user_id == 42
    assert account.account_label == "main"


def test_hh_account_all_fields_are_correct(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    _create_hh_account(database)

    database.update_hh_account(
        42,
        last_checked_at="2026-07-26T10:00:00+00:00",
        last_error="",
    )
    # Legacy rows are created outside the public API on purpose: the whole
    # point of Phase 5.6 is that no public writer can create ``connected``.
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            """
            UPDATE hh_accounts
            SET status = ?, last_login_at = ?
            WHERE telegram_user_id = ?
            """,
            ("connected", "2026-07-26T09:30:00+00:00", 42),
        )

    row = database.get_hh_account(42)
    assert row is not None
    account = Database._hh_account(row)
    assert account is not None

    assert account.id == row["id"]
    assert account.telegram_user_id == 42
    assert account.account_label == "main"
    # Phase 5.6: legacy values are normalised on read without rewriting
    # the stored row.
    assert account.status == "authorized"
    assert account.browser_profile_dir == "/tmp/profile"
    assert account.last_login_at == "2026-07-26T09:30:00+00:00"
    assert account.last_checked_at == "2026-07-26T10:00:00+00:00"
    assert account.last_error == ""
    assert account.created_at == row["created_at"]
    assert account.updated_at == row["updated_at"]


def test_hh_account_optional_fields_default_to_none(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    _create_hh_account(database)

    row = database.get_hh_account(42)
    assert row is not None
    account = Database._hh_account(row)
    assert account is not None

    # last_login_at / last_checked_at are NULL until first login/check
    assert account.last_login_at is None
    assert account.last_checked_at is None


def test_hh_account_from_none_row_returns_none(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.init()

    assert Database._hh_account(None) is None
    assert Database._hh_account(database.get_hh_account(42)) is None


def test_hh_account_is_immutable(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    _create_hh_account(database)

    row = database.get_hh_account(42)
    assert row is not None
    account = Database._hh_account(row)
    assert account is not None

    with pytest.raises(dataclasses.FrozenInstanceError):
        account.account_label = "changed"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        account.status = "connected"  # type: ignore[misc]


def test_hh_account_does_not_break_upsert(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)

    account_id_1 = database.create_hh_account(
        telegram_user_id=42, account_label="main", now=NOW
    )
    account_id_2 = database.create_hh_account(
        telegram_user_id=42, account_label="updated", now=NOW
    )

    row = database.get_hh_account(42)
    assert row is not None
    account = Database._hh_account(row)
    assert account is not None

    assert account_id_1 == account_id_2
    assert account.account_label == "updated"


def test_hh_account_does_not_break_user_isolation(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)

    database.create_hh_account(telegram_user_id=42, account_label="alice", now=NOW)
    database.create_hh_account(telegram_user_id=99, account_label="bob", now=NOW)

    row_42 = database.get_hh_account(42)
    row_99 = database.get_hh_account(99)
    assert row_42 is not None
    assert row_99 is not None

    account_42 = Database._hh_account(row_42)
    account_99 = Database._hh_account(row_99)
    assert account_42 is not None
    assert account_99 is not None

    assert account_42.telegram_user_id == 42
    assert account_42.account_label == "alice"
    assert account_99.telegram_user_id == 99
    assert account_99.account_label == "bob"

    # User 99 update affects only its own account; user 42 remains unchanged
    assert database.update_hh_account(99, account_label="updated_bob")
    assert database.get_hh_account(42)["account_label"] == "alice"
    assert database.get_hh_account(99)["account_label"] == "updated_bob"

def test_get_or_create_hh_account_is_idempotent(tmp_path: Path) -> None:
    """Repeated calls preserve the existing HH account and its settings."""
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    first = database.get_or_create_hh_account(
        telegram_user_id=42,
        now=NOW,
    )
    assert database.update_hh_account(
        42,
        account_label="Primary account",
        browser_profile_dir="profiles/user_42",
    )
    assert database.confirm_hh_account_authorized(
        42, last_checked_at=NOW.isoformat()
    )
    second = database.get_or_create_hh_account(
        telegram_user_id=42,
        now=datetime(2026, 7, 27, 9, 0, tzinfo=UTC),
    )
    assert second.id == first.id
    assert second.telegram_user_id == 42
    assert second.account_label == "Primary account"
    assert second.browser_profile_dir == "profiles/user_42"
    assert second.status == "authorized"


# =============================================================================
# Phase 5.4: database-level HH account ownership lookup
# =============================================================================


def test_get_hh_account_by_id_requires_matching_owner(tmp_path: Path) -> None:
    """Both the account id and Telegram user id must own the same row."""
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)
    account_id = database.create_hh_account(
        telegram_user_id=42,
        account_label="alice",
        browser_profile_dir="profiles/alice",
        now=NOW,
    )

    account = database.get_hh_account_by_id(account_id, 42)

    assert account is not None
    assert isinstance(account, HHAccount)
    assert account.id == account_id
    assert account.telegram_user_id == 42
    assert account.browser_profile_dir == "profiles/alice"


def test_get_hh_account_by_id_rejects_wrong_telegram_user(tmp_path: Path) -> None:
    """A valid account id must not leak an account to another user."""
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)
    account_id = database.create_hh_account(
        telegram_user_id=42,
        account_label="alice",
        now=NOW,
    )

    assert database.get_hh_account_by_id(account_id, 99) is None


def test_get_hh_account_by_id_rejects_nonexistent_account(tmp_path: Path) -> None:
    """A nonexistent account id never returns another user's row."""
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_hh_account(
        telegram_user_id=42,
        account_label="alice",
        now=NOW,
    )

    assert database.get_hh_account_by_id(999_999, 42) is None


@pytest.mark.parametrize(
    "account_id",
    [None, 0, -1, "", "not-an-id", True, False],
)
def test_get_hh_account_by_id_handles_invalid_ids_safely(
    tmp_path: Path, account_id: object
) -> None:
    """Invalid ids are rejected without raising or matching a row."""
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_hh_account(
        telegram_user_id=42,
        account_label="alice",
        now=NOW,
    )

    assert database.get_hh_account_by_id(account_id, 42) is None  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "lookalike",
    [
        1.0,
        1.9,
        2.0,
        "1",
        "1.0",
        "1.9",
        " 1 ",
        "+1",
        "01",
        b"1",
        2**63,
        10**100,
    ],
)
def test_get_hh_account_by_id_rejects_non_integer_or_out_of_range_ids(
    tmp_path: Path, lookalike: object
) -> None:
    """float/str/bytes lookalikes must never be truncated into a valid id."""
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    account_id = database.create_hh_account(
        telegram_user_id=42,
        account_label="alice",
        now=NOW,
    )

    # Both identifier positions are validated independently.
    assert database.get_hh_account_by_id(lookalike, 42) is None
    assert database.get_hh_account_by_id(account_id, lookalike) is None


# =============================================================================
# Phase 5.5: persisted HH account authorisation state
# =============================================================================


def test_update_hh_account_persists_authorization_state(tmp_path: Path) -> None:
    """Existing state columns round-trip through the typed model."""
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_hh_account(
        telegram_user_id=42,
        account_label="alice",
        now=NOW,
    )

    assert database.confirm_hh_account_authorized(
        42,
        last_login_at="2026-07-30T12:00:00+00:00",
        last_checked_at="2026-07-30T12:05:00+00:00",
        last_error="",
    )

    account = database.get_hh_account_model(42)
    assert account is not None
    assert account.status == "authorized"
    assert account.last_login_at == "2026-07-30T12:00:00+00:00"
    assert account.last_checked_at == "2026-07-30T12:05:00+00:00"
    assert account.last_error == ""


def test_update_hh_account_clears_stale_error_and_is_user_scoped(
    tmp_path: Path,
) -> None:
    """A successful check can clear last_error; missing users write nothing."""
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=99, now=NOW)
    database.create_hh_account(
        telegram_user_id=42,
        account_label="alice",
        now=NOW,
    )
    database.create_hh_account(
        telegram_user_id=99,
        account_label="bob",
        now=NOW,
    )
    assert database.transition_hh_account(
        42,
        "login_required",
        last_error="HH.kz authorization is required",
    )

    assert database.confirm_hh_account_authorized(
        42, last_checked_at=NOW.isoformat(), last_error=""
    )

    account_42 = database.get_hh_account_model(42)
    account_99 = database.get_hh_account_model(99)
    assert account_42 is not None
    assert account_99 is not None
    assert account_42.status == "authorized"
    assert account_42.last_error == ""
    assert account_99.status == "disconnected"
    assert account_99.last_error == ""

    assert database.confirm_hh_account_authorized(
        123456, last_checked_at=NOW.isoformat()
    ) is False


# ============================================================================
# Phase 5.6 - HH account state machine and legacy read normalisation
# ============================================================================


def _phase56_database(tmp_path: Path) -> Database:
    database = make_database(tmp_path)
    database.create_user(telegram_user_id=42, now=NOW)
    database.get_or_create_hh_account(telegram_user_id=42, now=NOW)
    return database


def test_hh_account_status_helpers() -> None:
    assert HHAccountStatus.SESSION_EXPIRED.value == "session_expired"
    assert normalize_hh_account_status("connected") == "authorized"
    assert normalize_hh_account_status("unknown") == "unknown"
    assert is_hh_account_authorized("connected") is True
    assert is_hh_account_authorized("authorized") is True
    assert is_hh_account_authorized("session_expired") is False
    assert is_hh_account_authorized("check_failed") is False


def test_hh_account_transition_table_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="Unknown HH account status"):
        validate_hh_account_transition("authorized", "totally-new")
    with pytest.raises(ValueError, match="Unknown HH account status"):
        validate_hh_account_transition("authorized", None)


def test_hh_account_transition_table_blocks_illegal_jump() -> None:
    with pytest.raises(ValueError, match="Illegal HH account status transition"):
        validate_hh_account_transition("disconnected", "session_expired")
    with pytest.raises(ValueError, match="Illegal HH account status transition"):
        validate_hh_account_transition("login_required", "logged_out")
    assert HH_ACCOUNT_TRANSITIONS["authorized"] >= {
        "session_expired",
        "logged_out",
        "check_failed",
        "unknown",
    }


def test_update_hh_account_rejects_all_authorization_fields(
    tmp_path: Path,
) -> None:
    """Ordinary updates cannot assign legacy, authorized or login time."""
    database = _phase56_database(tmp_path)

    for fields in (
        {"status": "connected"},
        {"status": "authorized"},
        {"status": "totally-new"},
        {"last_login_at": NOW.isoformat()},
    ):
        with pytest.raises(ValueError, match="centralised HH"):
            database.update_hh_account(42, **fields)

    account = database.get_hh_account_model(42)
    assert account is not None
    assert account.status == "disconnected"
    assert account.last_login_at is None


def test_authorized_requires_the_centralized_confirmation_path(
    tmp_path: Path,
) -> None:
    database = _phase56_database(tmp_path)

    with pytest.raises(ValueError, match="AUTHORIZED requires"):
        database.transition_hh_account(42, "authorized")
    with pytest.raises(ValueError, match="legacy read-only"):
        database.transition_hh_account(42, "connected")
    with pytest.raises(ValueError, match="Unknown HH account status"):
        database.transition_hh_account(42, "totally-new")
    with pytest.raises(ValueError, match="Illegal HH account status transition"):
        database.transition_hh_account(42, "session_expired")

    assert database.confirm_hh_account_authorized(
        42, last_checked_at=NOW.isoformat()
    )
    account = database.get_hh_account_model(42)
    assert account is not None
    assert account.status == "authorized"


def test_legacy_connected_is_normalised_on_read_only(tmp_path: Path) -> None:
    database = _phase56_database(tmp_path)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE hh_accounts SET status = ? WHERE telegram_user_id = ?",
            ("connected", 42),
        )

    account = database.get_hh_account_model(42)
    assert account is not None
    assert account.status == "authorized"

    with sqlite3.connect(database.path) as connection:
        stored = connection.execute(
            "SELECT status FROM hh_accounts WHERE telegram_user_id = ?",
            (42,),
        ).fetchone()
    assert stored is not None
    assert stored[0] == "connected"


def test_repeated_checks_after_logged_out_and_expired_can_reauthorize(
    tmp_path: Path,
) -> None:
    database = _phase56_database(tmp_path)
    assert database.confirm_hh_account_authorized(
        42,
        last_login_at=NOW.isoformat(),
        last_checked_at=NOW.isoformat(),
        last_error="",
    )
    assert database.transition_hh_account(42, "session_expired")
    assert database.transition_hh_account(42, "check_failed")
    assert database.confirm_hh_account_authorized(
        42, last_checked_at=NOW.isoformat()
    )

    assert database.transition_hh_account(42, "logged_out")
    assert database.transition_hh_account(42, "unknown")
    assert database.confirm_hh_account_authorized(
        42, last_checked_at=NOW.isoformat(), last_error=""
    )

    account = database.get_hh_account_model(42)
    assert account is not None
    assert account.status == "authorized"


def test_transition_hh_account_updates_persisted_check_metadata(
    tmp_path: Path,
) -> None:
    database = _phase56_database(tmp_path)
    checked_at = (NOW + timedelta(minutes=5)).isoformat()

    assert database.transition_hh_account(
        42,
        "check_failed",
        last_checked_at=checked_at,
        last_error="HH.kz session check failed due to a technical error",
    )

    account = database.get_hh_account_model(42)
    assert account is not None
    assert account.status == "check_failed"
    assert account.last_checked_at == checked_at
    assert account.last_error == (
        "HH.kz session check failed due to a technical error"
    )


def test_hh_account_status_isolation_between_accounts(tmp_path: Path) -> None:
    database = _phase56_database(tmp_path)
    database.create_user(telegram_user_id=43, now=NOW)
    database.get_or_create_hh_account(telegram_user_id=43, now=NOW)

    assert database.confirm_hh_account_authorized(
        42,
        last_login_at=NOW.isoformat(),
        last_checked_at=NOW.isoformat(),
    )
    assert database.transition_hh_account(42, "logged_out")

    account_42 = database.get_hh_account_model(42)
    account_43 = database.get_hh_account_model(43)
    assert account_42 is not None
    assert account_43 is not None
    assert account_42.status == "logged_out"
    assert account_43.status == "disconnected"


def test_no_phase56_schema_migration_is_required(tmp_path: Path) -> None:
    database = _phase56_database(tmp_path)
    with sqlite3.connect(database.path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(hh_accounts)")
        }

    assert columns == {
        "id",
        "telegram_user_id",
        "account_label",
        "status",
        "browser_profile_dir",
        "last_login_at",
        "last_checked_at",
        "last_error",
        "created_at",
        "updated_at",
    }


class _RecordingConnection:
    """Record executed SQL while delegating to a real connection.

    Used to prove that the transition opens its transaction BEFORE it
    reads the current status.
    """

    def __init__(
        self, connection: sqlite3.Connection, statements: list[str]
    ) -> None:
        self._connection = connection
        self._statements = statements

    def execute(self, sql: str, parameters=()):
        self._statements.append(sql.strip().upper())
        return self._connection.execute(sql, parameters)

    def __getattr__(self, name: str):
        return getattr(self._connection, name)

    def __enter__(self) -> "_RecordingConnection":
        self._connection.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._connection.__exit__(*exc_info)


def test_hh_account_transition_locks_before_reading_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BEGIN IMMEDIATE must precede the SELECT, validate and UPDATE."""
    database = _phase56_database(tmp_path)
    statements: list[str] = []
    original_connect = Database._connect

    def recording_connect(self: Database) -> _RecordingConnection:
        return _RecordingConnection(original_connect(self), statements)

    monkeypatch.setattr(Database, "_connect", recording_connect)

    assert database.transition_hh_account(
        42, "check_failed", last_checked_at=NOW.isoformat()
    )

    begin_index = statements.index("BEGIN IMMEDIATE")
    select_index = next(
        index
        for index, sql in enumerate(statements)
        if sql.startswith("SELECT STATUS FROM HH_ACCOUNTS")
    )
    update_index = next(
        index
        for index, sql in enumerate(statements)
        if sql.startswith("UPDATE HH_ACCOUNTS")
    )
    assert begin_index < select_index < update_index
    # No write may happen before the transaction is opened.
    assert not any(
        sql.startswith(("UPDATE", "INSERT", "DELETE"))
        for sql in statements[:begin_index]
    )


def test_hh_account_transition_ignores_stale_concurrent_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer cannot change the status between the read and the write."""
    database = _phase56_database(tmp_path)
    account = database.get_hh_account_model(42)
    assert account is not None
    assert account.status == "disconnected"

    competing_outcome: list[str] = []
    real_validate = validate_hh_account_transition

    def racing_validate(current: str, target: object) -> str:
        # Runs INSIDE the transition, after BEGIN IMMEDIATE and the SELECT.
        try:
            with sqlite3.connect(database.path, timeout=0.2) as other:
                other.execute(
                    "UPDATE hh_accounts SET status = ? "
                    "WHERE telegram_user_id = ?",
                    ("logged_out", 42),
                )
            competing_outcome.append("applied")
        except sqlite3.OperationalError:
            competing_outcome.append("blocked")
        return real_validate(current, target)

    monkeypatch.setattr(
        "database.validate_hh_account_transition", racing_validate
    )

    assert database.transition_hh_account(
        42, "check_failed", last_checked_at=NOW.isoformat()
    )

    assert competing_outcome == ["blocked"]
    stored = database.get_hh_account_model(42)
    assert stored is not None
    # The transition was validated against the state that was still
    # current while the write lock was held.
    assert stored.status == "check_failed"


def test_hh_account_transition_rolls_back_and_releases_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A validation error rolls back and leaves no lock or partial write."""
    database = _phase56_database(tmp_path)
    assert database.confirm_hh_account_authorized(
        42, last_checked_at=NOW.isoformat(), last_error=""
    )

    def exploding_validate(current: str, target: object) -> str:
        raise RuntimeError("validation exploded")

    monkeypatch.setattr(
        "database.validate_hh_account_transition", exploding_validate
    )

    with pytest.raises(RuntimeError, match="validation exploded"):
        database.transition_hh_account(42, "logged_out")

    # The rollback released the write lock: another connection can write
    # immediately, and the failed transition changed nothing.
    with sqlite3.connect(database.path, timeout=0.2) as connection:
        connection.execute(
            "UPDATE hh_accounts SET last_error = ? WHERE telegram_user_id = ?",
            ("after-rollback", 42),
        )
    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == "authorized"
    assert stored.last_error == "after-rollback"
