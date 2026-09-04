from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Iterable

from config import CandidateProfile, HHProfile, Profile


class VacancyStatus(str, Enum):
    DISCOVERED = "discovered"
    REJECTED_BY_FILTER = "rejected_by_filter"
    PREVIEWED = "previewed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    APPLYING = "applying"
    SKIPPED = "skipped"
    APPLIED = "applied"
    APPLY_FAILED = "apply_failed"
    EXPIRED = "expired"


@dataclass(frozen=True)
class Vacancy:
    id: str
    title: str
    company: str
    company_url: str
    company_rating: float | None
    company_reviews_count: int | None
    salary: str
    schedule: str
    location: str
    experience: str
    url: str
    description_hash: str
    search_query: str
    status: VacancyStatus
    discovered_at: str
    approval_requested_at: str | None
    approval_expires_at: str | None
    approved_at: str | None
    applying_at: str | None
    submit_attempted_at: str | None
    applied_at: str | None
    approver_id: int | None
    error_text: str
    cover_letter: str


@dataclass(frozen=True)
class SearchRun:
    id: int
    started_at: str
    finished_at: str
    state: str
    query_count: int
    found_results: int
    new_vacancies: int
    duplicates: int
    rejected_by_filter: int
    telegram_cards: int
    error_count: int
    rejection_reasons: dict[str, int]
    error_reasons: dict[str, int]
    last_safe_error: str
    circuit_reason: str
    telegram_user_id: int | None = None


@dataclass(frozen=True)
class ClaimResult:
    allowed: bool
    reason: str
    vacancy: Vacancy | None = None


ALLOWED_TRANSITIONS = {
    VacancyStatus.DISCOVERED: {
        VacancyStatus.REJECTED_BY_FILTER,
        VacancyStatus.PREVIEWED,
        VacancyStatus.PENDING_APPROVAL,
        VacancyStatus.APPLY_FAILED,
    },
    VacancyStatus.PREVIEWED: {
        VacancyStatus.PENDING_APPROVAL,
        VacancyStatus.APPLY_FAILED,
    },
    VacancyStatus.PENDING_APPROVAL: {
        VacancyStatus.APPROVED,
        VacancyStatus.SKIPPED,
        VacancyStatus.EXPIRED,
    },
    VacancyStatus.APPROVED: {
        VacancyStatus.APPLYING,
        VacancyStatus.EXPIRED,
        VacancyStatus.APPLY_FAILED,
    },
    VacancyStatus.APPLYING: {
        VacancyStatus.APPLIED,
        VacancyStatus.APPLY_FAILED,
    },
}


def _as_utc(value: datetime) -> datetime:
    return (value.replace(tzinfo=UTC) if value.tzinfo is None else value).astimezone(UTC)


def _iso(value: datetime) -> str:
    return _as_utc(value).isoformat()

class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        statuses = ", ".join(f"\'" + status.value + "\'" for status in VacancyStatus)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS applied_jobs (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    url TEXT,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    msg_id TEXT PRIMARY KEY,
                    chat_id TEXT,
                    text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS vacancies (
                    id TEXT PRIMARY KEY,
                    telegram_user_id INTEGER,
                    title TEXT NOT NULL,
                    company TEXT NOT NULL DEFAULT '',
                    company_url TEXT NOT NULL DEFAULT '',
                    company_rating REAL,
                    company_reviews_count INTEGER,
                    salary TEXT NOT NULL DEFAULT '',
                    schedule TEXT NOT NULL DEFAULT '',
                    location TEXT NOT NULL DEFAULT '',
                    experience TEXT NOT NULL DEFAULT '',
                    url TEXT NOT NULL,
                    description_hash TEXT NOT NULL DEFAULT '',
                    search_query TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL CHECK (status IN ({statuses})),
                    discovered_at TEXT NOT NULL,
                    approval_requested_at TEXT,
                    approval_expires_at TEXT,
                    approved_at TEXT,
                    applying_at TEXT,
                    submit_attempted_at TEXT,
                    applied_at TEXT,
                    approver_id INTEGER,
                    permit_hash TEXT,
                    error_text TEXT NOT NULL DEFAULT '',
                    cover_letter TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS vacancies_status_idx ON vacancies(status)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS vacancies_applied_at_idx ON vacancies(applied_at)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS search_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_user_id INTEGER,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (
                        state IN (\'completed\', \'failed\', \'paused_by_circuit_breaker\')
                    ),
                    query_count INTEGER NOT NULL DEFAULT 0,
                    found_results INTEGER NOT NULL DEFAULT 0,
                    new_vacancies INTEGER NOT NULL DEFAULT 0,
                    duplicates INTEGER NOT NULL DEFAULT 0,
                    rejected_by_filter INTEGER NOT NULL DEFAULT 0,
                    telegram_cards INTEGER NOT NULL DEFAULT 0,
                    error_count INTEGER NOT NULL DEFAULT 0,
                    rejection_reasons_json TEXT NOT NULL DEFAULT \'{}\',
                    error_reasons_json TEXT NOT NULL DEFAULT \'{}\',
                    last_safe_error TEXT NOT NULL DEFAULT \'\',
                    circuit_reason TEXT NOT NULL DEFAULT \'\',
                    FOREIGN KEY (telegram_user_id)
                        REFERENCES users(telegram_user_id)
                        ON DELETE SET NULL
                )
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(vacancies)")
            }
            migrations = {
                "applying_at": "TEXT",
                "submit_attempted_at": "TEXT",
                "company_url": "TEXT NOT NULL DEFAULT \'\'",
                "company_rating": "REAL",
                "company_reviews_count": "INTEGER",
                "salary": "TEXT NOT NULL DEFAULT \'\'",
                "schedule": "TEXT NOT NULL DEFAULT \'\'",
                "location": "TEXT NOT NULL DEFAULT \'\'",
                "experience": "TEXT NOT NULL DEFAULT \'\'",
                "cover_letter": "TEXT NOT NULL DEFAULT \'\'",
            }
            for name, definition in migrations.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE vacancies ADD COLUMN {name} {definition}"
                    )
            self._migrate_status_check(connection)
            self._migrate_search_runs_user_id(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    telegram_user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    created_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    telegram_user_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL DEFAULT \'\',
                    location TEXT NOT NULL DEFAULT \'\',
                    experience_summary TEXT NOT NULL DEFAULT \'\',
                    education TEXT NOT NULL DEFAULT \'\',
                    github_url TEXT NOT NULL DEFAULT \'\',
                    salary_expectation TEXT NOT NULL DEFAULT \'\',
                    additional_information TEXT NOT NULL DEFAULT \'\',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (telegram_user_id)
                        REFERENCES users(telegram_user_id)
                        ON DELETE CASCADE
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS search_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,

                    search_queries_json TEXT NOT NULL DEFAULT \'[]\',
                    areas_json TEXT NOT NULL DEFAULT \'[]\',
                    experience_filters_json TEXT NOT NULL DEFAULT \'[]\',
                    desired_positions_json TEXT NOT NULL DEFAULT \'[]\',
                    technologies_json TEXT NOT NULL DEFAULT \'[]\',
                    excluded_positions_json TEXT NOT NULL DEFAULT \'[]\',
                    work_format_json TEXT NOT NULL DEFAULT \'[]\',

                    min_salary INTEGER,

                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    FOREIGN KEY (telegram_user_id)
                        REFERENCES users(telegram_user_id)
                        ON DELETE CASCADE
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS search_profiles_user_idx
                ON search_profiles(telegram_user_id, enabled)
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS hh_accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_user_id INTEGER NOT NULL UNIQUE,

                    account_label TEXT NOT NULL DEFAULT \'\',
                    status TEXT NOT NULL DEFAULT \'disconnected\',

                    browser_profile_dir TEXT NOT NULL DEFAULT \'\',

                    last_login_at TEXT,
                    last_checked_at TEXT,
                    last_error TEXT NOT NULL DEFAULT \'\',

                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    FOREIGN KEY (telegram_user_id)
                        REFERENCES users(telegram_user_id)
                        ON DELETE CASCADE
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS resumes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_user_id INTEGER NOT NULL,

                    hh_resume_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT \'\',

                    is_active INTEGER NOT NULL DEFAULT 1,

                    last_raised_at TEXT,
                    next_raise_at TEXT,
                    raise_interval_minutes INTEGER,

                    last_error TEXT NOT NULL DEFAULT \'\',

                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    UNIQUE(telegram_user_id, hh_resume_id),

                    FOREIGN KEY (telegram_user_id)
                        REFERENCES users(telegram_user_id)
                        ON DELETE CASCADE
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS resumes_raise_idx
                ON resumes(is_active, next_raise_at)
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_vacancies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    telegram_user_id INTEGER NOT NULL,
                    vacancy_id TEXT NOT NULL,

                    search_profile_id INTEGER,

                    match_status TEXT NOT NULL DEFAULT \'matched\',

                    seen_at TEXT,
                    matched_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,

                    UNIQUE(telegram_user_id, vacancy_id),

                    FOREIGN KEY (telegram_user_id)
                        REFERENCES users(telegram_user_id)
                        ON DELETE CASCADE,

                    FOREIGN KEY (vacancy_id)
                        REFERENCES vacancies(id)
                        ON DELETE CASCADE,

                    FOREIGN KEY (search_profile_id)
                        REFERENCES search_profiles(id)
                        ON DELETE SET NULL
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS user_vacancies_user_idx
                ON user_vacancies(telegram_user_id, matched_at)
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS applications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    telegram_user_id INTEGER NOT NULL,
                    vacancy_id TEXT NOT NULL,
                    resume_id INTEGER,

                    status TEXT NOT NULL DEFAULT \'new\',

                    response_url TEXT NOT NULL DEFAULT \'\',
                    cover_letter TEXT NOT NULL DEFAULT \'\',
                    error_text TEXT NOT NULL DEFAULT \'\',

                    created_at TEXT NOT NULL,
                    applied_at TEXT,
                    updated_at TEXT NOT NULL,

                    UNIQUE(telegram_user_id, vacancy_id),

                    FOREIGN KEY (telegram_user_id)
                        REFERENCES users(telegram_user_id)
                        ON DELETE CASCADE,

                    FOREIGN KEY (vacancy_id)
                        REFERENCES vacancies(id)
                        ON DELETE CASCADE,

                    FOREIGN KEY (resume_id)
                        REFERENCES resumes(id)
                        ON DELETE SET NULL
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS applications_user_status_idx
                ON applications(
                    telegram_user_id,
                    status,
                    updated_at
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS application_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    application_id INTEGER NOT NULL,

                    status TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT \'\',

                    created_at TEXT NOT NULL,

                    FOREIGN KEY (application_id)
                        REFERENCES applications(id)
                        ON DELETE CASCADE
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    telegram_user_id INTEGER NOT NULL,

                    task_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT \'{}\',

                    status TEXT NOT NULL DEFAULT \'pending\',

                    run_at TEXT NOT NULL,

                    locked_at TEXT,
                    completed_at TEXT,

                    last_error TEXT NOT NULL DEFAULT \'\',

                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    FOREIGN KEY (telegram_user_id)
                        REFERENCES users(telegram_user_id)
                        ON DELETE CASCADE
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS scheduled_tasks_queue_idx
                ON scheduled_tasks(status, run_at)
                """
            )

            # Additional indexes for user-scoped queries
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS applications_user_idx
                ON applications(telegram_user_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS hh_accounts_user_idx
                ON hh_accounts(telegram_user_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS resumes_user_idx
                ON resumes(telegram_user_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS scheduled_tasks_user_idx
                ON scheduled_tasks(telegram_user_id, status)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS user_profiles_user_idx
                ON user_profiles(telegram_user_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS vacancies_user_idx
                ON vacancies(telegram_user_id)
                """
            )


    def create_user(
        self,
        telegram_user_id: int,
        username: str | None = None,
        first_name: str | None = None,
        now: datetime | None = None,
    ) -> bool:
        timestamp = _iso(now or datetime.now(UTC))

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO users (
                    telegram_user_id,
                    username,
                    first_name,
                    created_at,
                    is_active
                )
                VALUES (?, ?, ?, ?, 1)
                """,
                (
                    telegram_user_id,
                    username,
                    first_name,
                    timestamp,
                ),
            )

        return cursor.rowcount > 0

    def get_user(self, telegram_user_id: int) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    telegram_user_id,
                    username,
                    first_name,
                    created_at,
                    is_active
                FROM users
                WHERE telegram_user_id = ?
                """,
                (telegram_user_id,),
            ).fetchone()

    def update_user(
        self,
        telegram_user_id: int,
        username: str | None = None,
        first_name: str | None = None,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE users
                SET username = ?,
                    first_name = ?
                WHERE telegram_user_id = ?
                """,
                (
                    username,
                    first_name,
                    telegram_user_id,
                ),
            )

        return cursor.rowcount > 0

    def set_user_active(
        self,
        telegram_user_id: int,
        active: bool,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE users
                SET is_active = ?
                WHERE telegram_user_id = ?
                """,
                (
                    1 if active else 0,
                    telegram_user_id,
                ),
            )

        return cursor.rowcount > 0

    def get_active_users(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    telegram_user_id,
                    username,
                    first_name,
                    created_at,
                    is_active
                FROM users
                WHERE is_active = 1
                ORDER BY created_at
                """
            ).fetchall()

    def get_or_create_user(
        self,
        telegram_user_id: int,
        username: str | None = None,
        first_name: str | None = None,
        now: datetime | None = None,
    ) -> sqlite3.Row:
        user = self.get_user(telegram_user_id)

        if user is None:
            self.create_user(
                telegram_user_id=telegram_user_id,
                username=username,
                first_name=first_name,
                now=now,
            )
        else:
            self.update_user(
                telegram_user_id=telegram_user_id,
                username=username,
                first_name=first_name,
            )

        result = self.get_user(telegram_user_id)
        if result is None:
            raise RuntimeError("user could not be created or loaded")

        return result

    # ------------------------------------------------------------------
    # User Profile
    # ------------------------------------------------------------------

    def create_user_profile(
        self,
        telegram_user_id: int,
        name: str = "",
        location: str = "",
        experience_summary: str = "",
        education: str = "",
        github_url: str = "",
        salary_expectation: str = "",
        additional_information: str = "",
        now: datetime | None = None,
    ) -> bool:
        timestamp = _iso(now or datetime.now(UTC))
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO user_profiles (
                    telegram_user_id, name, location, experience_summary,
                    education, github_url, salary_expectation,
                    additional_information, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    telegram_user_id, name, location, experience_summary,
                    education, github_url, salary_expectation,
                    additional_information, timestamp, timestamp,
                ),
            )
            return cursor.rowcount == 1

    def get_user_profile(self, telegram_user_id: int) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT * FROM user_profiles
                WHERE telegram_user_id = ?
                """,
                (telegram_user_id,),
            ).fetchone()

    def update_user_profile(
        self,
        telegram_user_id: int,
        **fields: str,
    ) -> bool:
        allowed = {
            "name", "location", "experience_summary", "education",
            "github_url", "salary_expectation", "additional_information",
        }
        if unknown := set(fields) - allowed:
            raise ValueError(f"Unsupported profile fields: {', '.join(sorted(unknown))}")
        if not fields:
            return False
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = [*fields.values(), _iso(datetime.now(UTC)), telegram_user_id]
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE user_profiles
                SET {assignments}, updated_at = ?
                WHERE telegram_user_id = ?
                """,
                values,
            )
            return cursor.rowcount == 1

    def get_active_profile(self, telegram_user_id: int) -> Profile | None:
        """Собрать полный профиль пользователя из БД.

        Возвращает Profile(candidate, hh) если пользователь существует,
        иначе None.

        Активный search_profile -- первый enabled=1 по id ASC.
        Активное resume -- первое is_active=1 по id ASC.
        """
        with self._connect() as connection:
            # Проверяем, существует ли пользователь
            user_exists = connection.execute(
                "SELECT 1 FROM users WHERE telegram_user_id = ?",
                (telegram_user_id,),
            ).fetchone()
            if user_exists is None:
                return None

            # Загружаем user_profile
            user_profile = connection.execute(
                """
                SELECT name, location, experience_summary, education,
                       github_url, salary_expectation, additional_information
                FROM user_profiles
                WHERE telegram_user_id = ?
                """,
                (telegram_user_id,),
            ).fetchone()

            # Загружаем активный search_profile (первый enabled, по id ASC)
            search_profile = connection.execute(
                """
                SELECT search_queries_json, areas_json, experience_filters_json,
                       desired_positions_json, technologies_json,
                       excluded_positions_json, work_format_json
                FROM search_profiles
                WHERE telegram_user_id = ? AND enabled = 1
                ORDER BY id ASC
                LIMIT 1
                """,
                (telegram_user_id,),
            ).fetchone()

            # Загружаем активное резюме (первое is_active, по id ASC)
            resume = connection.execute(
                """
                SELECT title
                FROM resumes
                WHERE telegram_user_id = ? AND is_active = 1
                ORDER BY id ASC
                LIMIT 1
                """,
                (telegram_user_id,),
            ).fetchone()

        # Помощник для безопасной десериализации JSON
        def _load_json(field_value: str | None, default: list) -> list:
            if field_value is None:
                return default
            try:
                result = json.loads(field_value)
                return result if isinstance(result, list) else default
            except (json.JSONDecodeError, TypeError):
                return default

        # Формируем CandidateProfile
        candidate = CandidateProfile(
            name=user_profile["name"] if user_profile else "",
            location=user_profile["location"] if user_profile else "",
            desired_positions=tuple(_load_json(
                search_profile["desired_positions_json"] if search_profile else None, []
            )),
            experience_summary=user_profile["experience_summary"] if user_profile else "",
            education=user_profile["education"] if user_profile else "",
            technologies=tuple(_load_json(
                search_profile["technologies_json"] if search_profile else None, []
            )),
            projects=(),  # Поле projects отсутствует в схеме user_profiles
            github_url=user_profile["github_url"] if user_profile else "",
            salary_expectation=user_profile["salary_expectation"] if user_profile else "",
            work_format=tuple(_load_json(
                search_profile["work_format_json"] if search_profile else None, []
            )),
            excluded_positions=tuple(_load_json(
                search_profile["excluded_positions_json"] if search_profile else None, []
            )),
            additional_information=user_profile["additional_information"] if user_profile else "",
        )

        # Формируем HHProfile
        hh = HHProfile(
            resume_name=resume["title"] if resume else "",
            search_queries=tuple(_load_json(
                search_profile["search_queries_json"] if search_profile else None, []
            )),
            areas=tuple(_load_json(
                search_profile["areas_json"] if search_profile else None, []
            )),
            experience_filters=tuple(_load_json(
                search_profile["experience_filters_json"] if search_profile else None, []
            )),
        )

        return Profile(candidate=candidate, hh=hh)

    # ------------------------------------------------------------------
    # Search Profiles
    # ------------------------------------------------------------------

    def create_search_profile(
        self,
        telegram_user_id: int,
        name: str,
        search_queries: list[str] | None = None,
        areas: list[str] | None = None,
        experience_filters: list[str] | None = None,
        desired_positions: list[str] | None = None,
        technologies: list[str] | None = None,
        excluded_positions: list[str] | None = None,
        work_format: list[str] | None = None,
        min_salary: int | None = None,
        enabled: bool = True,
        now: datetime | None = None,
    ) -> int:
        timestamp = _iso(now or datetime.now(UTC))
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO search_profiles (
                    telegram_user_id, name, enabled,
                    search_queries_json, areas_json, experience_filters_json,
                    desired_positions_json, technologies_json,
                    excluded_positions_json, work_format_json,
                    min_salary, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    telegram_user_id,
                    name,
                    1 if enabled else 0,
                    json.dumps(search_queries or [], ensure_ascii=False),
                    json.dumps(areas or [], ensure_ascii=False),
                    json.dumps(experience_filters or [], ensure_ascii=False),
                    json.dumps(desired_positions or [], ensure_ascii=False),
                    json.dumps(technologies or [], ensure_ascii=False),
                    json.dumps(excluded_positions or [], ensure_ascii=False),
                    json.dumps(work_format or [], ensure_ascii=False),
                    min_salary,
                    timestamp,
                    timestamp,
                ),
            )
            return int(cursor.lastrowid)

    def get_search_profile(self, profile_id: int, telegram_user_id: int) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT * FROM search_profiles
                WHERE id = ? AND telegram_user_id = ?
                """,
                (profile_id, telegram_user_id),
            ).fetchone()

    def get_search_profiles(self, telegram_user_id: int) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT * FROM search_profiles
                WHERE telegram_user_id = ?
                ORDER BY created_at
                """,
                (telegram_user_id,),
            ).fetchall()

    def update_search_profile(
        self,
        profile_id: int,
        telegram_user_id: int,
        **fields: object,
    ) -> bool:
        allowed = {
            "name", "enabled", "search_queries", "areas",
            "experience_filters", "desired_positions", "technologies",
            "excluded_positions", "work_format", "min_salary",
        }
        if unknown := set(fields) - allowed:
            raise ValueError(f"Unsupported search profile fields: {', '.join(sorted(unknown))}")
        if not fields:
            return False
        json_fields = {
            "search_queries", "areas", "experience_filters",
            "desired_positions", "technologies", "excluded_positions", "work_format",
        }
        assignments = []
        values = []
        for name, value in fields.items():
            if name == "enabled":
                assignments.append(f"{name} = ?")
                values.append(1 if value else 0)
            elif name in json_fields:
                assignments.append(f"{name}_json = ?")
                values.append(json.dumps(value, ensure_ascii=False))
            else:
                assignments.append(f"{name} = ?")
                values.append(value)
        values.extend([_iso(datetime.now(UTC)), profile_id, telegram_user_id])
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE search_profiles
                SET {', '.join(assignments)}, updated_at = ?
                WHERE id = ? AND telegram_user_id = ?
                """,
                values,
            )
            return cursor.rowcount == 1

    def delete_search_profile(self, profile_id: int, telegram_user_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM search_profiles
                WHERE id = ? AND telegram_user_id = ?
                """,
                (profile_id, telegram_user_id),
            )
            return cursor.rowcount == 1

    # ------------------------------------------------------------------
    # HH Accounts
    # ------------------------------------------------------------------

    def create_hh_account(
        self,
        telegram_user_id: int,
        account_label: str = "",
        browser_profile_dir: str = "",
        now: datetime | None = None,
    ) -> int:
        timestamp = _iso(now or datetime.now(UTC))
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO hh_accounts (
                    telegram_user_id, account_label, browser_profile_dir,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(telegram_user_id) DO UPDATE SET
                    account_label = excluded.account_label,
                    browser_profile_dir = excluded.browser_profile_dir,
                    updated_at = excluded.updated_at
                """,
                (
                    telegram_user_id,
                    account_label,
                    browser_profile_dir,
                    timestamp,
                    timestamp,
                ),
            )
            # For INSERT...ON CONFLICT, lastrowid may not be reliable
            row = connection.execute(
                "SELECT id FROM hh_accounts WHERE telegram_user_id = ?",
                (telegram_user_id,),
            ).fetchone()
            return int(row["id"]) if row else 0

    def get_hh_account(self, telegram_user_id: int) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT * FROM hh_accounts
                WHERE telegram_user_id = ?
                """,
                (telegram_user_id,),
            ).fetchone()

    def update_hh_account(
        self,
        telegram_user_id: int,
        **fields: object,
    ) -> bool:
        allowed = {
            "account_label", "status", "browser_profile_dir",
            "last_login_at", "last_checked_at", "last_error",
        }
        if unknown := set(fields) - allowed:
            raise ValueError(f"Unsupported hh_account fields: {', '.join(sorted(unknown))}")
        if not fields:
            return False
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = [*fields.values(), _iso(datetime.now(UTC)), telegram_user_id]
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE hh_accounts
                SET {assignments}, updated_at = ?
                WHERE telegram_user_id = ?
                """,
                values,
            )
            return cursor.rowcount == 1

    def delete_hh_account(self, telegram_user_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM hh_accounts
                WHERE telegram_user_id = ?
                """,
                (telegram_user_id,),
            )
            return cursor.rowcount == 1

    # ------------------------------------------------------------------
    # Resumes
    # ------------------------------------------------------------------

    def create_resume(
        self,
        telegram_user_id: int,
        hh_resume_id: str,
        title: str = "",
        is_active: bool = True,
        raise_interval_minutes: int | None = None,
        now: datetime | None = None,
    ) -> int:
        timestamp = _iso(now or datetime.now(UTC))
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO resumes (
                    telegram_user_id, hh_resume_id, title,
                    is_active, raise_interval_minutes,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    telegram_user_id,
                    hh_resume_id,
                    title,
                    1 if is_active else 0,
                    raise_interval_minutes,
                    timestamp,
                    timestamp,
                ),
            )
            return int(cursor.lastrowid)

    def get_resume(self, resume_id: int, telegram_user_id: int) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT * FROM resumes
                WHERE id = ? AND telegram_user_id = ?
                """,
                (resume_id, telegram_user_id),
            ).fetchone()

    def get_resumes(self, telegram_user_id: int) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT * FROM resumes
                WHERE telegram_user_id = ?
                ORDER BY created_at
                """,
                (telegram_user_id,),
            ).fetchall()

    def update_resume(
        self,
        resume_id: int,
        telegram_user_id: int,
        **fields: object,
    ) -> bool:
        allowed = {
            "title", "is_active", "last_raised_at", "next_raise_at",
            "raise_interval_minutes", "last_error",
        }
        if unknown := set(fields) - allowed:
            raise ValueError(f"Unsupported resume fields: {', '.join(sorted(unknown))}")
        if not fields:
            return False
        assignments = []
        values = []
        for name, value in fields.items():
            if name == "is_active":
                assignments.append(f"{name} = ?")
                values.append(1 if value else 0)
            else:
                assignments.append(f"{name} = ?")
                values.append(value)
        values.extend([_iso(datetime.now(UTC)), resume_id, telegram_user_id])
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE resumes
                SET {', '.join(assignments)}, updated_at = ?
                WHERE id = ? AND telegram_user_id = ?
                """,
                values,
            )
            return cursor.rowcount == 1

    def delete_resume(self, resume_id: int, telegram_user_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM resumes
                WHERE id = ? AND telegram_user_id = ?
                """,
                (resume_id, telegram_user_id),
            )
            return cursor.rowcount == 1

    # ------------------------------------------------------------------
    # Applications
    # ------------------------------------------------------------------

    def create_application(
        self,
        telegram_user_id: int,
        vacancy_id: str,
        resume_id: int | None = None,
        cover_letter: str = "",
        now: datetime | None = None,
    ) -> int:
        timestamp = _iso(now or datetime.now(UTC))
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO applications (
                    telegram_user_id, vacancy_id, resume_id,
                    cover_letter, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    telegram_user_id,
                    vacancy_id,
                    resume_id,
                    cover_letter,
                    timestamp,
                    timestamp,
                ),
            )
            return int(cursor.lastrowid) if cursor.rowcount == 1 else 0

    def get_application(self, application_id: int, telegram_user_id: int) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT * FROM applications
                WHERE id = ? AND telegram_user_id = ?
                """,
                (application_id, telegram_user_id),
            ).fetchone()

    def get_applications(self, telegram_user_id: int) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT * FROM applications
                WHERE telegram_user_id = ?
                ORDER BY created_at DESC
                """,
                (telegram_user_id,),
            ).fetchall()

    def update_application(
        self,
        application_id: int,
        telegram_user_id: int,
        **fields: object,
    ) -> bool:
        allowed = {
            "status", "resume_id", "response_url", "cover_letter",
            "error_text", "applied_at",
        }
        if unknown := set(fields) - allowed:
            raise ValueError(f"Unsupported application fields: {', '.join(sorted(unknown))}")
        if not fields:
            return False
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = [*fields.values(), _iso(datetime.now(UTC)), application_id, telegram_user_id]
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE applications
                SET {assignments}, updated_at = ?
                WHERE id = ? AND telegram_user_id = ?
                """,
                values,
            )
            return cursor.rowcount == 1

    def add_application_event(
        self,
        application_id: int,
        telegram_user_id: int,
        status: str,
        message: str = "",
        now: datetime | None = None,
    ) -> int:
        timestamp = _iso(now or datetime.now(UTC))
        with self._connect() as connection:
            # Verify ownership before adding event
            row = connection.execute(
                """
                SELECT 1 FROM applications
                WHERE id = ? AND telegram_user_id = ?
                """,
                (application_id, telegram_user_id),
            ).fetchone()
            if row is None:
                return 0
            cursor = connection.execute(
                """
                INSERT INTO application_events (
                    application_id, status, message, created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (application_id, status, message, timestamp),
            )
            return int(cursor.lastrowid)

    def get_application_events(
        self,
        application_id: int,
        telegram_user_id: int,
    ) -> list[sqlite3.Row]:
        with self._connect() as connection:
            # Verify ownership
            row = connection.execute(
                """
                SELECT 1 FROM applications
                WHERE id = ? AND telegram_user_id = ?
                """,
                (application_id, telegram_user_id),
            ).fetchone()
            if row is None:
                return []
            return connection.execute(
                """
                SELECT * FROM application_events
                WHERE application_id = ?
                ORDER BY created_at
                """,
                (application_id,),
            ).fetchall()

    # ------------------------------------------------------------------
    # Scheduled Tasks
    # ------------------------------------------------------------------

    def create_scheduled_task(
        self,
        telegram_user_id: int,
        task_type: str,
        payload: dict[str, object] | None = None,
        run_at: datetime | None = None,
        now: datetime | None = None,
    ) -> int:
        timestamp = _iso(now or datetime.now(UTC))
        run_timestamp = _iso(run_at or datetime.now(UTC))
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO scheduled_tasks (
                    telegram_user_id, task_type, payload_json,
                    run_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    telegram_user_id,
                    task_type,
                    json.dumps(payload or {}, ensure_ascii=False),
                    run_timestamp,
                    timestamp,
                    timestamp,
                ),
            )
            return int(cursor.lastrowid)

    def get_scheduled_task(
        self,
        task_id: int,
        telegram_user_id: int,
    ) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT * FROM scheduled_tasks
                WHERE id = ? AND telegram_user_id = ?
                """,
                (task_id, telegram_user_id),
            ).fetchone()

    def get_scheduled_tasks(
        self,
        telegram_user_id: int,
        status: str | None = None,
    ) -> list[sqlite3.Row]:
        with self._connect() as connection:
            if status:
                return connection.execute(
                    """
                    SELECT * FROM scheduled_tasks
                    WHERE telegram_user_id = ? AND status = ?
                    ORDER BY run_at
                    """,
                    (telegram_user_id, status),
                ).fetchall()
            return connection.execute(
                """
                SELECT * FROM scheduled_tasks
                WHERE telegram_user_id = ?
                ORDER BY run_at
                """,
                (telegram_user_id,),
            ).fetchall()

    def update_scheduled_task(
        self,
        task_id: int,
        telegram_user_id: int,
        **fields: object,
    ) -> bool:
        allowed = {"status", "payload", "run_at", "locked_at", "completed_at", "last_error"}
        if unknown := set(fields) - allowed:
            raise ValueError(f"Unsupported scheduled_task fields: {', '.join(sorted(unknown))}")
        if not fields:
            return False
        assignments = []
        values = []
        for name, value in fields.items():
            if name == "payload":
                assignments.append("payload_json = ?")
                values.append(json.dumps(value, ensure_ascii=False))
            elif name in ("run_at", "locked_at", "completed_at") and isinstance(value, datetime):
                assignments.append(f"{name} = ?")
                values.append(_iso(value))
            else:
                assignments.append(f"{name} = ?")
                values.append(value)
        values.extend([_iso(datetime.now(UTC)), task_id, telegram_user_id])
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE scheduled_tasks
                SET {', '.join(assignments)}, updated_at = ?
                WHERE id = ? AND telegram_user_id = ?
                """,
                values,
            )
            return cursor.rowcount == 1

    def delete_scheduled_task(self, task_id: int, telegram_user_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM scheduled_tasks
                WHERE id = ? AND telegram_user_id = ?
                """,
                (task_id, telegram_user_id),
            )
            return cursor.rowcount == 1

    def _migrate_search_runs_user_id(self, connection: sqlite3.Connection) -> None:
        """Добавить telegram_user_id в search_runs если столбец отсутствует."""
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(search_runs)")
        }
        if "telegram_user_id" not in columns:
            connection.execute(
                "ALTER TABLE search_runs ADD COLUMN telegram_user_id INTEGER"
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS search_runs_user_idx
                ON search_runs(telegram_user_id, started_at)
                """
            )

    def _migrate_status_check(self, connection: sqlite3.Connection) -> None:
        """Пересоздать таблицу vacancies если CHECK-ограничение не включает новые статусы.

        SQLite не позволяет изменить CHECK-ограничение через ALTER TABLE --
        поэтому при появлении новых значений VacancyStatus нужна полная пересборка таблицы.
        """
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type=\'table\' AND name=\'vacancies\'"
        ).fetchone()
        if row is None:
            return
        table_sql: str = row["sql"] or ""
        missing = [s.value for s in VacancyStatus if f"\'{s.value}\'" not in table_sql]
        if not missing:
            return
        import logging as _logging
        _logging.getLogger(__name__).info(
            "db_migrate_status_check missing=%s", missing
        )
        statuses = ", ".join(f"\'{s.value}\'" for s in VacancyStatus)
        connection.execute("ALTER TABLE vacancies RENAME TO _vacancies_old")
        connection.execute(
            f"""
            CREATE TABLE vacancies (
                id TEXT PRIMARY KEY,
                telegram_user_id INTEGER,
                title TEXT NOT NULL,
                company TEXT NOT NULL DEFAULT \'\',
                company_url TEXT NOT NULL DEFAULT \'\',
                company_rating REAL,
                company_reviews_count INTEGER,
                salary TEXT NOT NULL DEFAULT \'\',
                schedule TEXT NOT NULL DEFAULT \'\',
                location TEXT NOT NULL DEFAULT \'\',
                experience TEXT NOT NULL DEFAULT \'\',
                url TEXT NOT NULL,
                description_hash TEXT NOT NULL DEFAULT \'\',
                search_query TEXT NOT NULL DEFAULT \'\',
                status TEXT NOT NULL CHECK (status IN ({statuses})),
                discovered_at TEXT NOT NULL,
                approval_requested_at TEXT,
                approval_expires_at TEXT,
                approved_at TEXT,
                applying_at TEXT,
                submit_attempted_at TEXT,
                applied_at TEXT,
                approver_id INTEGER,
                permit_hash TEXT,
                error_text TEXT NOT NULL DEFAULT \'\',
                cover_letter TEXT NOT NULL DEFAULT \'\'
            )
            """
        )
        old_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(_vacancies_old)"
            ).fetchall()
        }

        def select_column(name: str, default_sql: str = "NULL") -> str:
            if name in old_columns:
                return name
            return f"{default_sql} AS {name}"

        insert_columns = (
            "id, telegram_user_id, title, company, company_url, company_rating, "
            "company_reviews_count, salary, schedule, location, experience, "
            "url, description_hash, search_query, status, discovered_at, "
            "approval_requested_at, approval_expires_at, approved_at, "
            "applying_at, submit_attempted_at, applied_at, approver_id, "
            "permit_hash, error_text, cover_letter"
        )

        select_columns = ", ".join(
            (
                select_column("id"),
                select_column("telegram_user_id"),
                select_column("title"),
                select_column("company", "\'\'"),
                select_column("company_url", "\'\'"),
                select_column("company_rating"),
                select_column("company_reviews_count"),
                select_column("salary", "\'\'"),
                select_column("schedule", "\'\'"),
                select_column("location", "\'\'"),
                select_column("experience", "\'\'"),
                select_column("url"),
                select_column("description_hash", "\'\'"),
                select_column("search_query", "\'\'"),
                select_column("status"),
                select_column("discovered_at"),
                select_column("approval_requested_at"),
                select_column("approval_expires_at"),
                select_column("approved_at"),
                select_column("applying_at"),
                select_column("submit_attempted_at"),
                select_column("applied_at"),
                select_column("approver_id"),
                select_column("permit_hash"),
                select_column("error_text", "\'\'"),
                select_column("cover_letter", "\'\'"),
            )
        )

        connection.execute(
            f"""
            INSERT INTO vacancies ({insert_columns})
            SELECT {select_columns}
            FROM _vacancies_old
            """
        )
        connection.execute("DROP TABLE _vacancies_old")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS vacancies_status_idx ON vacancies(status)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS vacancies_applied_at_idx ON vacancies(applied_at)"
        )

    @staticmethod
    def _vacancy(row: sqlite3.Row | None) -> Vacancy | None:
        if row is None:
            return None
        return Vacancy(
            id=row["id"],
            title=row["title"],
            company=row["company"],
            company_url=row["company_url"],
            company_rating=(
                float(row["company_rating"])
                if row["company_rating"] is not None
                else None
            ),
            company_reviews_count=(
                int(row["company_reviews_count"])
                if row["company_reviews_count"] is not None
                else None
            ),
            salary=row["salary"],
            schedule=row["schedule"],
            location=row["location"],
            experience=row["experience"],
            url=row["url"],
            description_hash=row["description_hash"],
            search_query=row["search_query"],
            status=VacancyStatus(row["status"]),
            discovered_at=row["discovered_at"],
            approval_requested_at=row["approval_requested_at"],
            approval_expires_at=row["approval_expires_at"],
            approved_at=row["approved_at"],
            applying_at=row["applying_at"],
            submit_attempted_at=row["submit_attempted_at"],
            applied_at=row["applied_at"],
            approver_id=row["approver_id"],
            error_text=row["error_text"],
            cover_letter=row["cover_letter"],
        )

    def discover(
        self,
        *,
        job_id: str,
        title: str,
        company: str,
        telegram_user_id: int | None = None,
        company_url: str = "",
        url: str,
        description_hash: str,
        search_query: str,
        discovered_at: datetime,
        salary: str = "",
        schedule: str = "",
        location: str = "",
        experience: str = "",
        company_rating: float | None = None,
        company_reviews_count: int | None = None,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO vacancies (
                    id, telegram_user_id, title, company, company_url, salary, schedule, location, experience,
                    company_rating, company_reviews_count,
                    url, description_hash, search_query, status, discovered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    telegram_user_id,
                    title,
                    company,
                    company_url,
                    salary,
                    schedule,
                    location,
                    experience,
                    company_rating,
                    company_reviews_count,
                    url,
                    description_hash,
                    search_query,
                    VacancyStatus.DISCOVERED.value,
                    _iso(discovered_at),
                ),
            )
            return cursor.rowcount == 1

    def store_company_details(
        self, job_id: str, *, rating: float | None, reviews_count: int | None, telegram_user_id: int | None = None
    ) -> bool:
        with self._connect() as connection:
            if telegram_user_id is not None:
                cursor = connection.execute(
                    """
                    UPDATE vacancies
                    SET company_rating = ?, company_reviews_count = ?
                    WHERE id = ? AND (telegram_user_id = ? OR telegram_user_id IS NULL)
                    """,
                    (rating, reviews_count, job_id, telegram_user_id),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE vacancies
                    SET company_rating = ?, company_reviews_count = ?
                    WHERE id = ?
                    """,
                    (rating, reviews_count, job_id),
                )
            return cursor.rowcount == 1

    def store_vacancy_details(
        self,
        job_id: str,
        *,
        salary: str,
        schedule: str,
        location: str,
        experience: str,
        telegram_user_id: int | None = None,
    ) -> bool:
        with self._connect() as connection:
            if telegram_user_id is not None:
                cursor = connection.execute(
                    """
                    UPDATE vacancies
                    SET salary = ?,
                        schedule = ?,
                        location = ?,
                        experience = ?
                    WHERE id = ? AND (telegram_user_id = ? OR telegram_user_id IS NULL)
                    """,
                    (salary, schedule, location, experience, job_id, telegram_user_id),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE vacancies
                    SET salary = ?,
                        schedule = ?,
                        location = ?,
                        experience = ?
                    WHERE id = ?
                    """,
                    (salary, schedule, location, experience, job_id),
                )
            return cursor.rowcount == 1

    def get(self, job_id: str, telegram_user_id: int | None = None) -> Vacancy | None:
        with self._connect() as connection:
            if telegram_user_id is not None:
                row = connection.execute(
                    "SELECT * FROM vacancies WHERE id = ? AND (telegram_user_id = ? OR telegram_user_id IS NULL)",
                    (job_id, telegram_user_id),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM vacancies WHERE id = ?", (job_id,)
                ).fetchone()
            return self._vacancy(row)

    def unprocessed_discovered(self, telegram_user_id: int | None = None) -> list[Vacancy]:
        with self._connect() as connection:
            if telegram_user_id is not None:
                rows = connection.execute(
                    """
                    SELECT * FROM vacancies
                    WHERE status = ? AND (telegram_user_id = ? OR telegram_user_id IS NULL)
                    ORDER BY discovered_at
                    """,
                    (VacancyStatus.DISCOVERED.value, telegram_user_id),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM vacancies
                    WHERE status = ?
                    ORDER BY discovered_at
                    """,
                    (VacancyStatus.DISCOVERED.value,),
                ).fetchall()
            return [self._vacancy(row) for row in rows]

    def transition(
        self,
        job_id: str,
        expected: VacancyStatus | Iterable[VacancyStatus],
        target: VacancyStatus,
        telegram_user_id: int | None = None,
        **fields: object,
    ) -> bool:
        expected_statuses = (
            (expected,) if isinstance(expected, VacancyStatus) else tuple(expected)
        )
        if not expected_statuses or any(
            target not in ALLOWED_TRANSITIONS.get(status, set()) for status in expected_statuses
        ):
            return False
        allowed_fields = {"error_text"}
        if unknown := set(fields) - allowed_fields:
            raise ValueError(f"Unsupported transition fields: {', '.join(sorted(unknown))}")
        assignments = ["status = ?", *(f"{name} = ?" for name in fields)]
        values = [target.value, *fields.values(), job_id, *(status.value for status in expected_statuses)]
        placeholders = ", ".join("?" for _ in expected_statuses)
        with self._connect() as connection:
            if telegram_user_id is not None:
                query = f"UPDATE vacancies SET {', '.join(assignments)} WHERE id = ? AND status IN ({placeholders}) AND (telegram_user_id = ? OR telegram_user_id IS NULL)"
                values = [*values, telegram_user_id]
            else:
                query = f"UPDATE vacancies SET {', '.join(assignments)} WHERE id = ? AND status IN ({placeholders})"
            cursor = connection.execute(query, values)
            return cursor.rowcount == 1

    def request_approval(
        self,
        *,
        job_id: str,
        cover_letter: str = "",
        now: datetime,
        telegram_user_id: int | None = None,
    ) -> bool:
        with self._connect() as connection:
            if telegram_user_id is not None:
                cursor = connection.execute(
                    """
                    UPDATE vacancies SET
                        status = ?, approval_requested_at = ?,
                        approval_expires_at = NULL, error_text = \'\', cover_letter = ?
                    WHERE id = ? AND status = ? AND (telegram_user_id = ? OR telegram_user_id IS NULL)
                    """,
                    (
                        VacancyStatus.PENDING_APPROVAL.value,
                        _iso(now),
                        cover_letter,
                        job_id,
                        VacancyStatus.DISCOVERED.value,
                        telegram_user_id,
                    ),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE vacancies SET
                        status = ?, approval_requested_at = ?,
                        approval_expires_at = NULL, error_text = \'\', cover_letter = ?
                    WHERE id = ? AND status = ?
                    """,
                    (
                        VacancyStatus.PENDING_APPROVAL.value,
                        _iso(now),
                        cover_letter,
                        job_id,
                        VacancyStatus.DISCOVERED.value,
                    ),
                )
            return cursor.rowcount == 1

    def approve(
        self,
        job_id: str,
        telegram_user_id: int,
        expected_user_id: int,
        now: datetime,
        ttl_minutes: int = 30,
    ) -> str | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM vacancies WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None or row["status"] != VacancyStatus.PENDING_APPROVAL.value:
                return None
            if telegram_user_id != expected_user_id:
                return None
            if row["telegram_user_id"] is not None and row["telegram_user_id"] != telegram_user_id:
                return None
            permit = secrets.token_urlsafe(32)
            permit_hash = hashlib.sha256(permit.encode()).hexdigest()
            cursor = connection.execute(
                """
                UPDATE vacancies SET
                    status = ?, approved_at = ?, approval_expires_at = ?,
                    approver_id = ?, permit_hash = ?
                WHERE id = ? AND status = ?
                """,
                (
                    VacancyStatus.APPROVED.value,
                    _iso(now),
                    _iso(now + timedelta(minutes=ttl_minutes)),
                    telegram_user_id,
                    permit_hash,
                    job_id,
                    VacancyStatus.PENDING_APPROVAL.value,
                ),
            )
            return permit if cursor.rowcount == 1 else None

    def skip(self, job_id: str, telegram_user_id: int, expected_user_id: int) -> bool:
        if telegram_user_id != expected_user_id:
            return False
            if row["telegram_user_id"] is not None and row["telegram_user_id"] != telegram_user_id:
                return None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE vacancies SET status = ?, approver_id = ? WHERE id = ? AND status = ?",
                (
                    VacancyStatus.SKIPPED.value,
                    telegram_user_id,
                    job_id,
                    VacancyStatus.PENDING_APPROVAL.value,
                ),
            )
            return cursor.rowcount == 1

    def claim_application(
        self,
        *,
        job_id: str,
        permit: str,
        telegram_user_id: int,
        expected_user_id: int,
        app_mode: str,
        enable_real_apply: bool,
        daily_limit: int,
        now: datetime,
    ) -> ClaimResult:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM vacancies WHERE id = ?", (job_id,)
            ).fetchone()
            vacancy = self._vacancy(row)
            if app_mode != "approval":
                return ClaimResult(False, "app_mode", vacancy)
            if not enable_real_apply:
                return ClaimResult(False, "real_apply_disabled", vacancy)
            if row is None or row["status"] != VacancyStatus.APPROVED.value:
                return ClaimResult(False, "invalid_status", vacancy)
            if not row["approval_requested_at"] or not row["approved_at"]:
                return ClaimResult(False, "missing_approval", vacancy)
            if telegram_user_id != expected_user_id or row["approver_id"] != expected_user_id:
                return ClaimResult(False, "wrong_user", vacancy)
            if not row["approval_expires_at"] or row["approval_expires_at"] <= _iso(now):
                connection.execute(
                    "UPDATE vacancies SET status = ?, permit_hash = NULL WHERE id = ? AND status = ?",
                    (VacancyStatus.EXPIRED.value, job_id, VacancyStatus.APPROVED.value),
                )
                return ClaimResult(False, "expired", vacancy)
            supplied_hash = hashlib.sha256(permit.encode()).hexdigest()
            if not secrets.compare_digest(row["permit_hash"] or "", supplied_hash):
                return ClaimResult(False, "invalid_permit", vacancy)
            if self._reserved_today(connection, now, telegram_user_id) >= daily_limit:
                return ClaimResult(False, "daily_limit", vacancy)
            cursor = connection.execute(
                "UPDATE vacancies SET status = ?, applying_at = ? WHERE id = ? AND status = ? AND permit_hash = ?",
                (
                    VacancyStatus.APPLYING.value,
                    _iso(now),
                    job_id,
                    VacancyStatus.APPROVED.value,
                    row["permit_hash"],
                ),
            )
            claimed = cursor.rowcount == 1
            return ClaimResult(
                claimed,
                "allowed" if claimed else "concurrent_claim",
                vacancy,
            )

    def complete_application(
        self,
        job_id: str,
        permit: str,
        *,
        success: bool,
        now: datetime,
        error_text: str = "",
    ) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            supplied_hash = hashlib.sha256(permit.encode()).hexdigest()
            cursor = connection.execute(
                """
                UPDATE vacancies SET status = ?, applied_at = ?, error_text = ?, permit_hash = NULL
                WHERE id = ? AND status = ? AND permit_hash = ?
                """,
                (
                    (VacancyStatus.APPLIED if success else VacancyStatus.APPLY_FAILED).value,
                    _iso(now) if success else None,
                    "" if success else error_text,
                    job_id,
                    VacancyStatus.APPLYING.value,
                    supplied_hash,
                ),
            )
            return cursor.rowcount == 1

    def mark_submit_attempt(
        self,
        job_id: str,
        permit: str,
        *,
        telegram_user_id: int | None = None,
        now: datetime,
        daily_limit: int,
    ) -> bool:
        supplied_hash = hashlib.sha256(permit.encode()).hexdigest()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM vacancies WHERE id = ?", (job_id,)
            ).fetchone()
            if (
                row is None
                or row["status"] != VacancyStatus.APPLYING.value
                or not secrets.compare_digest(row["permit_hash"] or "", supplied_hash)
            ):
                return False
            if not row["approval_expires_at"] or row["approval_expires_at"] <= _iso(now):
                connection.execute(
                    "UPDATE vacancies SET status = ?, permit_hash = NULL WHERE id = ? AND status = ? AND permit_hash = ?",
                    (
                        VacancyStatus.EXPIRED.value,
                        job_id,
                        VacancyStatus.APPLYING.value,
                        row["permit_hash"],
                    ),
                )
                return False
            start, end = self._day_bounds(now)
            this_row_reserved = any(
                timestamp is not None and start <= timestamp < end
                for timestamp in (row["applying_at"], row["submit_attempted_at"])
            )
            reserved = self._reserved_today(connection, now, telegram_user_id) if telegram_user_id is not None else self._reserved_today(connection, now)
            if reserved - int(this_row_reserved) >= daily_limit:
                return False
            cursor = connection.execute(
                """
                UPDATE vacancies SET submit_attempted_at = ?
                WHERE id = ? AND status = ? AND permit_hash = ?
                    AND submit_attempted_at IS NULL
                """,
                (
                    _iso(now),
                    job_id,
                    VacancyStatus.APPLYING.value,
                    row["permit_hash"],
                ),
            )
            return cursor.rowcount == 1

    @staticmethod
    def _day_bounds(now: datetime) -> tuple[str, str]:
        aware = now.replace(tzinfo=UTC) if now.tzinfo is None else now
        start = datetime.combine(aware.date(), time.min, tzinfo=aware.tzinfo)
        return _iso(start), _iso(start + timedelta(days=1))

    def _applied_today(self, connection: sqlite3.Connection, now: datetime, telegram_user_id: int | None = None) -> int:
        start, end = self._day_bounds(now)
        if telegram_user_id is not None:
            return int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM vacancies
                    WHERE (
                        telegram_user_id = ?
                        OR (telegram_user_id IS NULL AND approver_id = ?)
                    )
                    AND status = ? AND applied_at >= ? AND applied_at < ?
                    """,
                    (telegram_user_id, telegram_user_id, VacancyStatus.APPLIED.value, start, end),
                ).fetchone()[0]
            )
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM vacancies WHERE status = ? AND applied_at >= ? AND applied_at < ?",
                (VacancyStatus.APPLIED.value, start, end),
            ).fetchone()[0]
        )

    def _reserved_today(self, connection: sqlite3.Connection, now: datetime, telegram_user_id: int | None = None) -> int:
        start, end = self._day_bounds(now)
        if telegram_user_id is not None:
            return int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM vacancies
                    WHERE (
                        telegram_user_id = ?
                        OR (telegram_user_id IS NULL AND approver_id = ?)
                    )
                    AND (
                        (status = ? AND applied_at >= ? AND applied_at < ?)
                        OR (status = ? AND applying_at >= ? AND applying_at < ?)
                        OR (status IN (?, ?) AND submit_attempted_at >= ? AND submit_attempted_at < ?)
                    )
                    """,
                    (
                        telegram_user_id,
                        telegram_user_id,
                        VacancyStatus.APPLIED.value,
                        start,
                        end,
                        VacancyStatus.APPLYING.value,
                        start,
                        end,
                        VacancyStatus.APPLYING.value,
                        VacancyStatus.APPLY_FAILED.value,
                        start,
                        end,
                    ),
                ).fetchone()[0]
            )
        return int(
            connection.execute(
                """
                SELECT COUNT(*) FROM vacancies
                WHERE (status = ? AND applied_at >= ? AND applied_at < ?)
                   OR (status = ? AND applying_at >= ? AND applying_at < ?)
                   OR (status IN (?, ?) AND submit_attempted_at >= ? AND submit_attempted_at < ?)
                """,
                (
                    VacancyStatus.APPLIED.value,
                    start,
                    end,
                    VacancyStatus.APPLYING.value,
                    start,
                    end,
                    VacancyStatus.APPLYING.value,
                    VacancyStatus.APPLY_FAILED.value,
                    start,
                    end,
                ),
            ).fetchone()[0]
        )

    def applied_today(self, now: datetime, telegram_user_id: int | None = None) -> int:
        with self._connect() as connection:
            return self._applied_today(connection, now, telegram_user_id)

    def expire_approved(self, now: datetime, telegram_user_id: int | None = None) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if telegram_user_id is not None:
                cursor = connection.execute(
                    """
                    UPDATE vacancies SET status = ?, permit_hash = NULL
                    WHERE status = ?
                        AND approval_expires_at IS NOT NULL
                        AND approval_expires_at <= ?
                        AND (telegram_user_id = ? OR telegram_user_id IS NULL)
                    """,
                    (
                        VacancyStatus.EXPIRED.value,
                        VacancyStatus.APPROVED.value,
                        _iso(now),
                        telegram_user_id,
                    ),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE vacancies SET status = ?, permit_hash = NULL
                    WHERE status = ?
                        AND approval_expires_at IS NOT NULL
                        AND approval_expires_at <= ?
                    """,
                    (
                        VacancyStatus.EXPIRED.value,
                        VacancyStatus.APPROVED.value,
                        _iso(now),
                    ),
                )
            return cursor.rowcount

    def pending(self, telegram_user_id: int | None = None) -> list[Vacancy]:
        with self._connect() as connection:
            if telegram_user_id is not None:
                rows = connection.execute(
                    "SELECT * FROM vacancies WHERE status = ? AND (telegram_user_id = ? OR telegram_user_id IS NULL) ORDER BY discovered_at",
                    (VacancyStatus.PENDING_APPROVAL.value, telegram_user_id),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM vacancies WHERE status = ? ORDER BY discovered_at",
                    (VacancyStatus.PENDING_APPROVAL.value,),
                ).fetchall()
            return [self._vacancy(row) for row in rows]

    def stats(self, telegram_user_id: int | None = None) -> dict[str, int]:
        counts = {status.value: 0 for status in VacancyStatus}
        with self._connect() as connection:
            if telegram_user_id is not None:
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM vacancies WHERE telegram_user_id = ? GROUP BY status",
                    (telegram_user_id,),
                ):
                    counts[row["status"]] = row["count"]
            else:
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM vacancies GROUP BY status"
                ):
                    counts[row["status"]] = row["count"]
        return counts

    def save_search_run(
        self,
        *,
        started_at: datetime,
        finished_at: datetime,
        state: str,
        query_count: int,
        found_results: int,
        new_vacancies: int,
        duplicates: int,
        rejected_by_filter: int,
        telegram_cards: int,
        error_count: int,
        rejection_reasons: dict[str, int],
        error_reasons: dict[str, int],
        last_safe_error: str = "",
        circuit_reason: str = "",
        telegram_user_id: int | None = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO search_runs (
                    telegram_user_id, started_at, finished_at, state, query_count, found_results,
                    new_vacancies, duplicates, rejected_by_filter,
                    telegram_cards, error_count, rejection_reasons_json,
                    error_reasons_json, last_safe_error, circuit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    telegram_user_id,
                    _iso(started_at),
                    _iso(finished_at),
                    state,
                    query_count,
                    found_results,
                    new_vacancies,
                    duplicates,
                    rejected_by_filter,
                    telegram_cards,
                    error_count,
                    json.dumps(rejection_reasons, ensure_ascii=False, sort_keys=True),
                    json.dumps(error_reasons, ensure_ascii=False, sort_keys=True),
                    last_safe_error,
                    circuit_reason,
                ),
            )
            return int(cursor.lastrowid)

    def latest_search_run(self, telegram_user_id: int | None = None) -> SearchRun | None:
        with self._connect() as connection:
            if telegram_user_id is not None:
                row = connection.execute(
                    """
                    SELECT * FROM search_runs
                    WHERE telegram_user_id = ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (telegram_user_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM search_runs ORDER BY id DESC LIMIT 1"
                ).fetchone()
        if row is None:
            return None
        return SearchRun(
            id=row["id"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            state=row["state"],
            query_count=row["query_count"],
            found_results=row["found_results"],
            new_vacancies=row["new_vacancies"],
            duplicates=row["duplicates"],
            rejected_by_filter=row["rejected_by_filter"],
            telegram_cards=row["telegram_cards"],
            error_count=row["error_count"],
            rejection_reasons=json.loads(row["rejection_reasons_json"]),
            error_reasons=json.loads(row["error_reasons_json"]),
            last_safe_error=row["last_safe_error"],
            circuit_reason=row["circuit_reason"],
            telegram_user_id=row["telegram_user_id"],
        )

    def record_search_run_error(
        self, run_id: int, reason: str, telegram_user_id: int | None = None
    ) -> bool:
        with self._connect() as connection:
            if telegram_user_id is not None:
                row = connection.execute(
                    """
                    SELECT error_reasons_json FROM search_runs
                    WHERE id = ? AND telegram_user_id = ?
                    """,
                    (run_id, telegram_user_id),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT error_reasons_json FROM search_runs WHERE id = ?",
                    (run_id,),
                ).fetchone()
            if row is None:
                return False
            reasons = json.loads(row["error_reasons_json"])
            reasons[reason] = reasons.get(reason, 0) + 1
            if telegram_user_id is not None:
                cursor = connection.execute(
                    """
                    UPDATE search_runs
                    SET error_count = error_count + 1,
                        error_reasons_json = ?, last_safe_error = ?
                    WHERE id = ? AND telegram_user_id = ?
                    """,
                    (json.dumps(reasons, ensure_ascii=False, sort_keys=True), reason, run_id, telegram_user_id),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE search_runs
                    SET error_count = error_count + 1,
                        error_reasons_json = ?, last_safe_error = ?
                    WHERE id = ?
                    """,
                    (json.dumps(reasons, ensure_ascii=False, sort_keys=True), reason, run_id),
                )
            return cursor.rowcount == 1

    def is_message_processed(self, message_id: str) -> bool:
        with self._connect() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM chat_messages WHERE msg_id = ?", (message_id,)
                ).fetchone()
                is not None
            )

    def add_processed_message(self, message_id: str, chat_id: str, text_value: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO chat_messages (msg_id, chat_id, text) VALUES (?, ?, ?)",
                (message_id, chat_id, text_value),
            )
            return cursor.rowcount == 1