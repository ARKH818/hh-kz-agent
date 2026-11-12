"""Tests for Stage 4\2 — UserRuntimeContext.

Focused tests verifying the runtime context object itself.
Does NOT test integration with agent loop or Telegram.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from config import load_settings
from database import Database
from config import UserSettings, load_user_settings
from runtime_context import UserRuntimeContext, create_runtime_context

# Minimal environment for Settings — no global profile.yaml, no TG_USER_ID
_MINIMAL_ENV = {
    "TG_BOT_TOKEN": "fake-token",
    "APP_MODE": "dry_run",
    "ENABLE_REAL_APPLY": "false",
    "BROWSER_BACKEND": "playwright",
    "BROWSER_HEADLESS": "false",
}


def _make_settings(tmp_path: Path) -> object:
    """Create Settings with NO global profile.yaml."""
    from dataclasses import replace
    loaded = load_settings(
        profile_path=None,
        environ=_MINIMAL_ENV,
    )
    return replace(
        loaded,
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )


def _make_user_settings(settings) -> UserSettings:
    """Materialise the ONE per-user settings instance for a context."""
    return load_user_settings(
        settings,
        max_applications_per_day=10,
        approval_ttl_minutes=30,
    )


# ============================================================================
# Creation and basic fields
# ============================================================================

def test_context_created_with_telegram_user_id(tmp_path: Path) -> None:
    """Context is created with the correct telegram_user_id."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    ctx = create_runtime_context(telegram_user_id=42, settings=settings, database=database)

    assert ctx.telegram_user_id == 42


def test_context_holds_settings_and_database(tmp_path: Path) -> None:
    """Settings and Database instances are accessible via context."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    ctx = create_runtime_context(telegram_user_id=42, settings=settings, database=database)

    assert ctx.settings is settings
    assert ctx.database is database


# ============================================================================
# Default runtime state
# ============================================================================

def test_context_default_runtime_state(tmp_path: Path) -> None:
    """Fresh context has correct default runtime values."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    ctx = create_runtime_context(telegram_user_id=42, settings=settings, database=database)

    assert ctx.paused is False
    assert ctx.circuit_reason == ""
    assert ctx.consecutive_search_errors == 0
    assert ctx.next_run_at is None
    assert isinstance(ctx.wake_event, asyncio.Event)
    assert not ctx.wake_event.is_set()


def test_context_has_created_at(tmp_path: Path) -> None:
    """Context has a created_at timestamp."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    before = datetime.now().astimezone()
    ctx = create_runtime_context(telegram_user_id=42, settings=settings, database=database)
    after = datetime.now().astimezone()

    assert before <= ctx.created_at <= after


# ============================================================================
# Independence of multiple contexts
# ============================================================================

def test_two_contexts_have_different_user_ids(tmp_path: Path) -> None:
    """Two contexts for different users have different telegram_user_id."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    ctx_42 = create_runtime_context(telegram_user_id=42, settings=settings, database=database)
    ctx_43 = create_runtime_context(telegram_user_id=43, settings=settings, database=database)

    assert ctx_42.telegram_user_id == 42
    assert ctx_43.telegram_user_id == 43


def test_two_contexts_have_different_wake_events(tmp_path: Path) -> None:
    """Each context has its own asyncio.Event instance."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    ctx_42 = create_runtime_context(telegram_user_id=42, settings=settings, database=database)
    ctx_43 = create_runtime_context(telegram_user_id=43, settings=settings, database=database)

    assert ctx_42.wake_event is not ctx_43.wake_event


def test_paused_isolation_between_contexts(tmp_path: Path) -> None:
    """Changing paused on one context does not affect another."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    ctx_42 = create_runtime_context(telegram_user_id=42, settings=settings, database=database)
    ctx_43 = create_runtime_context(telegram_user_id=43, settings=settings, database=database)

    ctx_42.paused = True

    assert ctx_42.paused is True
    assert ctx_43.paused is False


def test_circuit_reason_isolation_between_contexts(tmp_path: Path) -> None:
    """Changing circuit_reason on one context does not affect another."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    ctx_42 = create_runtime_context(telegram_user_id=42, settings=settings, database=database)
    ctx_43 = create_runtime_context(telegram_user_id=43, settings=settings, database=database)

    ctx_42.circuit_reason = "page_structure_changed"

    assert ctx_42.circuit_reason == "page_structure_changed"
    assert ctx_43.circuit_reason == ""


def test_consecutive_search_errors_isolation(tmp_path: Path) -> None:
    """Changing consecutive_search_errors on one context does not affect another."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    ctx_42 = create_runtime_context(telegram_user_id=42, settings=settings, database=database)
    ctx_43 = create_runtime_context(telegram_user_id=43, settings=settings, database=database)

    ctx_42.consecutive_search_errors = 3

    assert ctx_42.consecutive_search_errors == 3
    assert ctx_43.consecutive_search_errors == 0


def test_wake_event_isolation(tmp_path: Path) -> None:
    """Setting one context's wake_event does not affect the other."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    ctx_42 = create_runtime_context(telegram_user_id=42, settings=settings, database=database)
    ctx_43 = create_runtime_context(telegram_user_id=43, settings=settings, database=database)

    ctx_42.wake_event.set()

    assert ctx_42.wake_event.is_set()
    assert not ctx_43.wake_event.is_set()


# ============================================================================
# Factory behaviour
# ============================================================================

def test_factory_creates_new_object_each_call(tmp_path: Path) -> None:
    """create_runtime_context returns a new object on every call."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    ctx_1 = create_runtime_context(telegram_user_id=42, settings=settings, database=database)
    ctx_2 = create_runtime_context(telegram_user_id=42, settings=settings, database=database)

    assert ctx_1 is not ctx_2
    assert ctx_1.telegram_user_id == ctx_2.telegram_user_id


# ============================================================================
# Stage 4\4 — Full isolation test between two runtime contexts
# ============================================================================

def test_two_contexts_fully_isolated(tmp_path: Path) -> None:
    """Two UserRuntimeContext objects are completely independent.

    Stage 4\4: verifies that per-user runtime control state (pause,
    circuit breaker, wake event) is not shared between users.
    """
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    ctx_a = create_runtime_context(telegram_user_id=42, settings=settings, database=database)
    ctx_b = create_runtime_context(telegram_user_id=43, settings=settings, database=database)

    # Different objects
    assert ctx_a is not ctx_b
    assert ctx_a.wake_event is not ctx_b.wake_event

    # --- paused isolation ---
    ctx_a.paused = True
    assert ctx_a.paused is True
    assert ctx_b.paused is False

    ctx_b.paused = True
    assert ctx_b.paused is True
    assert ctx_a.paused is True  # both paused independently

    # A.resume() must not affect B
    ctx_a.paused = False
    assert ctx_a.paused is False
    assert ctx_b.paused is True  # B still paused

    # --- circuit_reason isolation ---
    ctx_b.circuit_reason = "search_errors"
    assert ctx_b.circuit_reason == "search_errors"
    assert ctx_a.circuit_reason == ""

    ctx_a.circuit_reason = "page_structure_changed"
    assert ctx_a.circuit_reason == "page_structure_changed"
    assert ctx_b.circuit_reason == "search_errors"  # B unchanged

    # --- consecutive_search_errors isolation ---
    ctx_b.consecutive_search_errors = 5
    assert ctx_b.consecutive_search_errors == 5
    assert ctx_a.consecutive_search_errors == 0

    ctx_a.consecutive_search_errors = 2
    assert ctx_a.consecutive_search_errors == 2
    assert ctx_b.consecutive_search_errors == 5  # B unchanged

    # --- wake_event isolation ---
    ctx_a.wake_event.set()
    assert ctx_a.wake_event.is_set()
    assert not ctx_b.wake_event.is_set()

    ctx_a.wake_event.clear()
    ctx_b.wake_event.set()
    assert ctx_b.wake_event.is_set()
    assert not ctx_a.wake_event.is_set()

# ============================================================================
# Stage 4\6 — per-user settings inside runtime contexts
# ============================================================================

def test_context_user_settings_identity(tmp_path: Path) -> None:
    """The context stores the very same UserSettings object it was given."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    user_settings = _make_user_settings(settings)

    ctx = create_runtime_context(42, settings, database, user_settings)

    assert ctx.user_settings is user_settings


def test_context_without_user_settings_gets_fresh_snapshot(tmp_path: Path) -> None:
    """Legacy callers: factory creates a fresh snapshot per context."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    ctx_42 = create_runtime_context(42, settings, database)
    ctx_43 = create_runtime_context(43, settings, database)

    assert ctx_42.user_settings is not ctx_43.user_settings
    assert ctx_42.user_settings is not settings.user_defaults


def test_replace_isolation_between_users(tmp_path: Path) -> None:
    """dataclasses.replace() on user 42's settings never affects user 43."""
    from dataclasses import replace as dc_replace

    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    us_42 = _make_user_settings(settings)
    us_43 = _make_user_settings(settings)

    ctx_42 = create_runtime_context(42, settings, database, us_42)
    ctx_43 = create_runtime_context(43, settings, database, us_43)

    # Per-user change via immutable replace + rebind
    ctx_42.user_settings = dc_replace(ctx_42.user_settings, app_mode="approval")

    assert ctx_42.user_settings.app_mode == "approval"
    assert ctx_42.user_settings is not us_42  # new object
    assert ctx_43.user_settings.app_mode == "dry_run"
    assert ctx_43.user_settings is us_43  # untouched

# ============================================================================
# Phase 5.2 — HH account binding and isolation
# ============================================================================

def test_context_loads_own_hh_account(tmp_path: Path) -> None:
    """Each runtime context loads only its own Telegram user's HH account."""
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()

    database.create_user(42)
    database.create_user(43)

    account_42_id = database.create_hh_account(
        telegram_user_id=42,
        account_label="Account 42",
    )
    account_43_id = database.create_hh_account(
        telegram_user_id=43,
        account_label="Account 43",
    )

    ctx_42 = create_runtime_context(42, settings, database)
    ctx_43 = create_runtime_context(43, settings, database)

    assert ctx_42.hh_account is not None
    assert ctx_43.hh_account is not None
    assert ctx_42.hh_account.id == account_42_id
    assert ctx_43.hh_account.id == account_43_id
    assert ctx_42.hh_account.telegram_user_id == 42
    assert ctx_43.hh_account.telegram_user_id == 43
    assert ctx_42.hh_account.account_label == "Account 42"
