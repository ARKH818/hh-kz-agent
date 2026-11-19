import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.methods import SendRichMessage
from aiogram.types import InputRichBlockDetails, InputRichMessage

from config import load_settings
from database import Database
from hh_client import LoginState, LogoutResult
from config import load_user_settings
from runtime_context import create_runtime_context
from tests.test_config import VALID_ENV, write_profile
from tg_bot import TelegramService
from updater import ReleaseInfo


NOW = datetime(2026, 7, 26, 9, 0, tzinfo=UTC)


class FakeBot:
    def __init__(self, rich_error: Exception | None = None):
        self.messages: list[dict] = []
        self.rich_messages: list[dict] = []
        self.rich_error = rich_error

    async def send_rich_message(self, **kwargs):
        if self.rich_error:
            raise self.rich_error
        self.rich_messages.append(kwargs)

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)


class FakeUser:
    def __init__(self, user_id: int, username: str | None = None, first_name: str | None = None):
        self.id = user_id
        self.username = username
        self.first_name = first_name


class FakeMessage:
    def __init__(self, text: str, from_user: FakeUser):
        self.text = text
        self.from_user = from_user
        self.answer_text: str | None = None

    async def answer(self, text: str, **kwargs) -> None:
        self.answer_text = text


class FakeApprovalService:
    async def approve_and_apply(self, job_id: str, user_id: int):
        raise AssertionError("approval must not be called by command tests")

    def skip(self, job_id: str, user_id: int):
        raise AssertionError("skip must not be called by command tests")


def service(
    tmp_path: Path,
    app_mode: str = "dry_run",
    rich_error: Exception | None = None,
):
    settings = load_settings(
        profile_path=write_profile(tmp_path),
        environ={**VALID_ENV, "APP_MODE": app_mode},
    )
    settings = replace(settings, database_path=tmp_path / "agent.db")
    database = Database(settings.database_path)
    database.init()
    database.create_user(telegram_user_id=42, username="testuser42", now=NOW)
    context = create_runtime_context(42, settings, database, load_user_settings(settings))
    bot = FakeBot(rich_error)
    telegram = TelegramService(
        settings,
        database,
        FakeApprovalService(),
        contexts={42: context},
        bot=bot,
        now_factory=lambda: NOW,
    )
    return telegram, database, context, bot


def attach_confirmed_session(
    telegram: TelegramService, user_id: int = 42
) -> "FakeSessionClient":
    """Register a verified client for synchronous resume regression tests."""
    client = FakeSessionClient(user_id)
    telegram._hh_clients[user_id] = client
    context = telegram._contexts[user_id]
    context.auth_state = LoginState.AUTHORIZED.value
    context.auth_paused = False
    return client


def add_preview_vacancy(database: Database, telegram_user_id: int = 42) -> None:
    assert database.discover(
        job_id="job-1",
        title="Python <Developer>",
        company="Example & Co",
        company_url="https://hh.ru/employer/123",
        company_rating="4,7",
        company_reviews_count="128",
        url="https://example.com/vacancy/job-1",
        description_hash="hash",
        search_query="Python",
        discovered_at=NOW,
        experience="backend-сервисы.",
        telegram_user_id=telegram_user_id,
    )
    assert database.store_company_details("job-1", rating=4.7, reviews_count=128, telegram_user_id=telegram_user_id)
    assert database.request_approval(
        job_id="job-1",
        now=NOW,
        telegram_user_id=telegram_user_id,
    )


def add_search_run(database: Database, telegram_user_id: int = 42) -> None:
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
        circuit_reason="page_structure_changed",
        telegram_user_id=telegram_user_id,
    )


def test_foreign_user_cannot_pause_or_read_configuration(tmp_path: Path) -> None:
    telegram, _, context, _ = service(tmp_path)

    reply = telegram.command("pause", user_id=99)

    assert reply == "This bot is private."
    assert not context.paused
    assert "42" not in reply


def test_pause_resume_and_status_use_live_state(tmp_path: Path) -> None:
    telegram, _, context, _ = service(tmp_path)
    attach_confirmed_session(telegram, 42)

    assert telegram.command("pause", user_id=42) == "Agent paused."
    assert context.paused
    status = telegram.command("status", user_id=42)
    assert "mode: dry_run" in status
    assert "state: paused" in status
    assert telegram.command("resume", user_id=42) == "Agent resumed."
    assert not context.paused
    assert context.wake_event.is_set()


def test_diagnostics_shows_last_search_run(tmp_path: Path) -> None:
    telegram, database, context, _ = service(tmp_path)
    add_search_run(database)
    context.circuit_reason = "technical_failure_ratio"
    context.next_run_at = NOW + timedelta(minutes=30)

    reply = telegram.command("diagnostics", user_id=42)

    assert "длительность: 12 с" in reply
    assert "запросы: 2" in reply
    assert "найдено: 9" in reply
    assert "новые: 4" in reply
    assert "дубли: 5" in reply
    assert "фильтр: 2" in reply
    assert "title=2" in reply
    assert "карточки: 0" in reply
    assert "ошибки: 1" in reply
    assert "network_error" in reply
    assert "breaker: open (technical_failure_ratio)" in reply
    assert "следующий запуск:" in reply


def test_resume_clears_runtime_breaker_but_keeps_saved_diagnostics(
    tmp_path: Path,
) -> None:
    telegram, database, context, _ = service(tmp_path)
    attach_confirmed_session(telegram, 42)
    add_search_run(database)
    context.paused = True
    context.circuit_reason = "page_structure_changed"
    context.consecutive_search_errors = 3
    context.next_run_at = NOW + timedelta(minutes=30)

    assert telegram.command("resume", 42) == "Agent resumed."
    assert not context.paused
    assert context.circuit_reason == ""
    assert context.consecutive_search_errors == 0
    assert context.next_run_at is None
    assert context.wake_event.is_set()
    assert database.latest_search_run(telegram_user_id=42).circuit_reason == "page_structure_changed"
    diagnostics = telegram.command("diagnostics", 42)
    assert "breaker: closed" in diagnostics
    assert "last breaker: page_structure_changed" in diagnostics


def test_dry_run_preview_uses_rich_card_with_company_details(tmp_path: Path) -> None:
    telegram, database, _, bot = service(tmp_path, "dry_run")
    add_preview_vacancy(database)

    asyncio.run(telegram.send_preview(database.get("job-1", telegram_user_id=42), include_actions=False, telegram_user_id=42))

    assert bot.messages == []
    message = bot.rich_messages[0]
    assert message["reply_markup"] is None
    details = next(
        block
        for block in message["rich_message"].blocks
        if isinstance(block, InputRichBlockDetails)
        and block.summary == "Компания"
    )
    rendered = repr(details.blocks)
    assert "Example & Co" in rendered
    assert "★" not in rendered
    assert "отзывов" not in rendered
    assert "Опыт" in repr(message["rich_message"].blocks)


def rich_error(error_type: type[Exception], message: str) -> Exception:
    method = SendRichMessage(
        chat_id=42,
        rich_message=InputRichMessage(html="test"),
    )
    return error_type(method=method, message=message)


def test_unsupported_rich_message_falls_back_to_escaped_html(tmp_path: Path) -> None:
    telegram, database, _, bot = service(
        tmp_path,
        rich_error=rich_error(TelegramBadRequest, "method is not supported"),
    )
    add_preview_vacancy(database)

    asyncio.run(telegram.send_preview(database.get("job-1", telegram_user_id=42), include_actions=False, telegram_user_id=42))

    assert len(bot.messages) == 1
    assert "Python &lt;Developer&gt;" in bot.messages[0]["text"]
    assert "Example &amp; Co" in bot.messages[0]["text"]
    assert "<b>Опыт:</b> backend-сервисы." in bot.messages[0]["text"]


def test_transport_error_does_not_send_duplicate_fallback(tmp_path: Path) -> None:
    telegram, database, _, bot = service(
        tmp_path,
        rich_error=rich_error(TelegramNetworkError, "network disconnected"),
    )
    add_preview_vacancy(database)

    with pytest.raises(TelegramNetworkError):
        asyncio.run(telegram.send_preview(database.get("job-1", telegram_user_id=42), include_actions=False, telegram_user_id=42))

    assert bot.messages == []


def test_approval_preview_has_apply_and_skip_buttons(tmp_path: Path) -> None:
    telegram, database, _, bot = service(tmp_path, "approval")
    add_preview_vacancy(database)

    asyncio.run(telegram.send_preview(database.get("job-1", telegram_user_id=42), include_actions=True, telegram_user_id=42))

    buttons = bot.rich_messages[0]["reply_markup"].inline_keyboard[0]
    assert [button.text for button in buttons] == ["Откликнуться", "Пропустить"]
    assert [button.callback_data for button in buttons] == ["apply:job-1", "skip:job-1"]


def test_pending_and_stats_read_sqlite(tmp_path: Path) -> None:
    telegram, database, _, _ = service(tmp_path, "approval")
    add_preview_vacancy(database)

    assert "job-1" in telegram.command("pending", user_id=42)
    assert "pending_approval: 1" in telegram.command("stats", user_id=42)

# ========== Новые multi-user тесты (этап 3.5) ==========


def test_user_42_is_authorized_when_in_users_table(tmp_path: Path) -> None:
    """A. Пользователь 42 авторизован, если users содержит telegram_user_id=42."""
    telegram, database, _, _ = service(tmp_path)
    assert telegram.authorized(42) is True


def test_user_43_is_not_authorized_without_registration(tmp_path: Path) -> None:
    """Пользователь 43 не авторизован без регистрации."""
    telegram, database, _, _ = service(tmp_path)
    assert telegram.authorized(43) is False


def test_user_43_cannot_access_user_42_data(tmp_path: Path) -> None:
    """B. Пользователь 43 не может получить данные пользователя 42."""
    telegram, database, _, _ = service(tmp_path)
    add_preview_vacancy(database, telegram_user_id=42)
    add_search_run(database, telegram_user_id=42)

    # Пользователь 43 не авторизован
    assert telegram.command("status", user_id=43) == "This bot is private."
    assert telegram.command("pending", user_id=43) == "This bot is private."
    assert telegram.command("stats", user_id=43) == "This bot is private."
    assert telegram.command("diagnostics", user_id=43) == "This bot is private."


def test_stats_uses_telegram_user_id_42(tmp_path: Path) -> None:
    """C. /stats для пользователя 42 использует telegram_user_id=42."""
    telegram, database, _, _ = service(tmp_path, "approval")
    add_preview_vacancy(database, telegram_user_id=42)

    # Добавляем вакансию для пользователя 43
    database.create_user(telegram_user_id=43, username="testuser43", now=NOW)
    assert database.discover(
        job_id="job-2",
        title="Java Developer",
        company="Other Co",
        company_url="https://hh.ru/employer/456",
        url="https://example.com/vacancy/job-2",
        description_hash="hash2",
        search_query="Java",
        discovered_at=NOW,
        telegram_user_id=43,
    )

    stats_42 = telegram.command("stats", user_id=42)
    assert "pending_approval: 1" in stats_42
    # Не должно быть 2 вакансий
    assert "pending_approval: 2" not in stats_42


def test_pending_uses_telegram_user_id_42(tmp_path: Path) -> None:
    """D. /pending для пользователя 42 использует telegram_user_id=42."""
    telegram, database, _, _ = service(tmp_path, "approval")
    add_preview_vacancy(database, telegram_user_id=42)

    # Добавляем вакансию для пользователя 43
    database.create_user(telegram_user_id=43, username="testuser43", now=NOW)
    assert database.discover(
        job_id="job-2",
        title="Java Developer",
        company="Other Co",
        company_url="https://hh.ru/employer/456",
        url="https://example.com/vacancy/job-2",
        description_hash="hash2",
        search_query="Java",
        discovered_at=NOW,
        telegram_user_id=43,
    )
    database.request_approval(job_id="job-2", now=NOW, telegram_user_id=43)

    pending_42 = telegram.command("pending", user_id=42)
    assert "job-1" in pending_42
    assert "job-2" not in pending_42


def test_diagnostics_uses_user_scoped_latest_search_run(tmp_path: Path) -> None:
    """E. /diagnostics для пользователя 42 вызывает latest_search_run(telegram_user_id=42)."""
    telegram, database, _, _ = service(tmp_path)
    add_search_run(database, telegram_user_id=42)

    # Добавляем search_run для пользователя 43
    database.create_user(telegram_user_id=43, username="testuser43", now=NOW)
    database.save_search_run(
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=5),
        state="completed",
        query_count=1,
        found_results=3,
        new_vacancies=1,
        duplicates=2,
        rejected_by_filter=0,
        telegram_cards=1,
        error_count=0,
        rejection_reasons={},
        error_reasons={},
        last_safe_error="",
        circuit_reason="",
        telegram_user_id=43,
    )

    reply_42 = telegram.command("diagnostics", user_id=42)
    assert "запросы: 2" in reply_42  # Данные пользователя 42
    assert "найдено: 9" in reply_42


def test_notify_with_explicit_telegram_user_id(tmp_path: Path) -> None:
    """F. notify() с явным telegram_user_id отправляет сообщение нужному пользователю."""
    telegram, _, _, bot = service(tmp_path)

    asyncio.run(telegram.notify("Test message", telegram_user_id=42))

    assert len(bot.messages) == 1
    assert bot.messages[0]["chat_id"] == 42
    assert bot.messages[0]["text"] == "Test message"

# ========== Tests for Stage 4.1 — Real Telegram user registration ==========


def test_new_user_start_creates_user_in_db(tmp_path: Path) -> None:
    """New Telegram user calling /start gets registered in users table."""
    telegram, database, _, _ = service(tmp_path)

    # User 99 does not exist yet
    assert telegram.authorized(99) is False

    # Simulate /start from user 99
    user = FakeUser(user_id=99, username="john_doe", first_name="John")
    message = FakeMessage(text="/start", from_user=user)
    asyncio.run(telegram._command_handler(message))

    # User now exists and is active
    user_row = database.get_user(99)
    assert user_row is not None
    assert user_row["telegram_user_id"] == 99
    assert user_row["is_active"] == 1
    assert telegram.authorized(99) is True


def test_new_user_start_saves_metadata(tmp_path: Path) -> None:
    """Telegram username and first_name are saved during registration."""
    telegram, database, _, _ = service(tmp_path)

    user = FakeUser(user_id=99, username="john_doe", first_name="John")
    message = FakeMessage(text="/start", from_user=user)
    asyncio.run(telegram._command_handler(message))

    user_row = database.get_user(99)
    assert user_row["username"] == "john_doe"
    assert user_row["first_name"] == "John"


def test_repeated_start_does_not_create_duplicate(tmp_path: Path) -> None:
    """Calling /start twice does not create duplicate users."""
    telegram, database, _, _ = service(tmp_path)

    user = FakeUser(user_id=99, username="john_doe", first_name="John")
    message = FakeMessage(text="/start", from_user=user)

    asyncio.run(telegram._command_handler(message))
    asyncio.run(telegram._command_handler(message))

    # Count users with id 99
    with database._connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM users WHERE telegram_user_id = ?", (99,)
        ).fetchone()[0]
    assert count == 1


def test_new_user_start_creates_hh_account(tmp_path: Path) -> None:
    """A new Telegram user gets an HH account during /start."""
    telegram, database, _, _ = service(tmp_path)

    user = FakeUser(user_id=99, username="john_doe", first_name="John")
    message = FakeMessage(text="/start", from_user=user)
    asyncio.run(telegram._command_handler(message))

    account = database.get_hh_account_model(99)
    assert account is not None
    assert account.telegram_user_id == 99


def test_repeated_start_preserves_hh_account_settings(
    tmp_path: Path,
) -> None:
    """Repeated /start does not overwrite existing HH account settings."""
    telegram, database, _, _ = service(tmp_path)

    user = FakeUser(user_id=99, username="john_doe", first_name="John")
    message = FakeMessage(text="/start", from_user=user)
    asyncio.run(telegram._command_handler(message))

    account_before = database.get_hh_account_model(99)
    assert account_before is not None

    assert database.update_hh_account(
        telegram_user_id=99,
        account_label="Primary account",
        browser_profile_dir="profiles/user_99",
    )
    assert database.confirm_hh_account_authorized(
        99, last_checked_at=NOW.isoformat()
    )

    asyncio.run(telegram._command_handler(message))

    account_after = database.get_hh_account_model(99)
    assert account_after is not None
    assert account_after.id == account_before.id
    assert account_after.account_label == "Primary account"
    assert account_after.browser_profile_dir == "profiles/user_99"
    assert account_after.status == "authorized"
    assert account_after.last_checked_at == NOW.isoformat()


def test_start_updates_metadata_if_changed(tmp_path: Path) -> None:
    """Repeated /start updates username/first_name if they changed."""
    telegram, database, _, _ = service(tmp_path)

    # First registration
    user1 = FakeUser(user_id=99, username="old_name", first_name="Old")
    message1 = FakeMessage(text="/start", from_user=user1)
    asyncio.run(telegram._command_handler(message1))

    # Second registration with changed data
    user2 = FakeUser(user_id=99, username="new_name", first_name="New")
    message2 = FakeMessage(text="/start", from_user=user2)
    asyncio.run(telegram._command_handler(message2))

    user_row = database.get_user(99)
    assert user_row["username"] == "new_name"
    assert user_row["first_name"] == "New"


def test_unauthorized_user_cannot_use_protected_commands(tmp_path: Path) -> None:
    """Unregistered user cannot access protected commands."""
    telegram, _, _, _ = service(tmp_path)

    user = FakeUser(user_id=99)
    for cmd in ("status", "pending", "stats", "diagnostics", "pause", "resume"):
        message = FakeMessage(text=f"/{cmd}", from_user=user)
        asyncio.run(telegram._command_handler(message))
        assert message.answer_text == "This bot is private."


def test_after_start_user_can_use_commands(tmp_path: Path) -> None:
    """After /start, user is authorized and can use protected commands."""
    telegram, database, _, _ = service(tmp_path)

    # Register user 99
    user = FakeUser(user_id=99, username="john_doe", first_name="John")
    start_message = FakeMessage(text="/start", from_user=user)
    asyncio.run(telegram._command_handler(start_message))

    # Stage 4\4: create runtime context for the newly registered user
    context_99 = create_runtime_context(
        telegram_user_id=99,
        settings=telegram.settings,
        database=database,
    )
    telegram._contexts[99] = context_99

    # Now protected commands should work
    assert telegram.authorized(99) is True
    status_reply = telegram.command("status", user_id=99)
    assert status_reply != "This bot is private."
    assert "mode:" in status_reply

def test_registration_does_not_affect_existing_user(tmp_path: Path) -> None:
    """Registering a new user does not modify existing user 42 data."""
    telegram, database, _, _ = service(tmp_path)

    # Capture user 42 state before
    user_42_before = database.get_user(42)
    assert user_42_before is not None

    # Register user 99
    user = FakeUser(user_id=99, username="john_doe", first_name="John")
    message = FakeMessage(text="/start", from_user=user)
    asyncio.run(telegram._command_handler(message))

    # User 42 unchanged
    user_42_after = database.get_user(42)
    assert user_42_after["telegram_user_id"] == 42
    assert user_42_after["username"] == user_42_before["username"]
    assert user_42_after["is_active"] == 1


def test_existing_user_start_still_works(tmp_path: Path) -> None:
    """Existing user calling /start remains authorized and gets welcome back."""
    telegram, database, _, bot = service(tmp_path)

    # User 42 already exists from service() helper
    assert telegram.authorized(42) is True

    user = FakeUser(user_id=42, username="testuser42", first_name="Test")
    message = FakeMessage(text="/start", from_user=user)
    asyncio.run(telegram._command_handler(message))

    # Still authorized, no duplicate
    assert telegram.authorized(42) is True
    with database._connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM users WHERE telegram_user_id = ?", (42,)
        ).fetchone()[0]
    assert count == 1


# ========== Stage 4\3 focused tests — notify_update_available ==========


def test_notify_update_available_uses_explicit_telegram_user_id(tmp_path: Path) -> None:
    """notify_update_available sends notification to explicitly passed telegram_user_id."""
    telegram, _, _, bot = service(tmp_path)

    release = ReleaseInfo(tag_name="v2.0.0", html_url="https://example.com/release", name="v2.0.0", published_at="2026-09-01T00:00:00Z", body="Update notes")
    asyncio.run(telegram.notify_update_available(release, telegram_user_id=99))

    assert len(bot.messages) == 1
    assert bot.messages[0]["chat_id"] == 99
    assert "v2.0.0" in bot.messages[0]["text"]


def test_notify_update_available_does_not_use_settings_tg_user_id(tmp_path: Path) -> None:
    """notify_update_available does not depend on settings.tg_user_id (removed)."""
    telegram, _, _, _ = service(tmp_path)

    # settings no longer has tg_user_id attribute
    assert not hasattr(telegram.settings, "tg_user_id")


# ============================================================================
# Stage 4\4 — Per-user runtime control isolation via TelegramService
# ============================================================================


def test_pause_resume_isolated_between_two_users(tmp_path: Path) -> None:
    """Stage 4\4: /pause and /resume affect only the caller's own context."""
    settings = load_settings(
        profile_path=write_profile(tmp_path),
        environ={**VALID_ENV, "APP_MODE": "dry_run"},
    )
    settings = replace(settings, database_path=tmp_path / "agent.db")
    database = Database(settings.database_path)
    database.init()
    database.create_user(telegram_user_id=42, username="user42", now=NOW)
    database.create_user(telegram_user_id=43, username="user43", now=NOW)

    ctx_a = create_runtime_context(42, settings, database, load_user_settings(settings))
    ctx_b = create_runtime_context(43, settings, database, load_user_settings(settings))

    telegram = TelegramService(
        settings,
        database,
        FakeApprovalService(),
        contexts={42: ctx_a, 43: ctx_b},
        bot=FakeBot(),
        now_factory=lambda: NOW,
    )
    attach_confirmed_session(telegram, 42)
    attach_confirmed_session(telegram, 43)

    # 1. /pause from user 42 → only ctx_a paused
    assert telegram.command("pause", user_id=42) == "Agent paused."
    assert ctx_a.paused is True
    assert ctx_b.paused is False

    # 2. /status for each user reflects their own context
    status_a = telegram.command("status", user_id=42)
    assert "state: paused" in status_a
    status_b = telegram.command("status", user_id=43)
    assert "state: running" in status_b

    # 3. /resume from user 43 → ctx_b resumed, ctx_a untouched
    assert telegram.command("resume", user_id=43) == "Agent resumed."
    assert ctx_b.paused is False
    assert ctx_b.wake_event.is_set()
    assert ctx_a.paused is True
    assert not ctx_a.wake_event.is_set()

    # 4. /resume from user 42 → ctx_a resumed
    assert telegram.command("resume", user_id=42) == "Agent resumed."
    assert ctx_a.paused is False
    assert ctx_a.wake_event.is_set()


def test_pause_without_runtime_context_returns_safe_reply(tmp_path: Path) -> None:
    """Stage 4\4: /pause without a registered runtime context does not crash."""
    telegram, _, _, _ = service(tmp_path)

    # Remove the context to simulate polling-only mode
    telegram._contexts.clear()

    reply = telegram.command("pause", user_id=42)
    assert reply == "Agent runtime is not running for your account."

    reply = telegram.command("resume", user_id=42)
    assert reply == "Agent runtime is not running for your account."

    reply = telegram.command("status", user_id=42)
    assert reply == "Agent runtime is not running for your account."

    reply = telegram.command("diagnostics", user_id=42)
    assert "runtime not running" in reply

# ============================================================================
# Stage 4\6 — /status shows the calling user's own app_mode
# ============================================================================

def test_status_shows_user_specific_mode(tmp_path: Path) -> None:
    """/status reflects the calling user's own app_mode, not a global one."""
    from dataclasses import replace as dc_replace

    telegram, database, context, _ = service(tmp_path, "dry_run")
    # User 42 switches their own runtime to approval mode (immutable replace)
    context.user_settings = dc_replace(context.user_settings, app_mode="approval")

    status = telegram.command("status", user_id=42)
    assert "mode: approval" in status

# ============================================================================
# Stage 4\7 — CAPTCHA futures are isolated per telegram_user_id
# ============================================================================


class FakeBotWithPhotos(FakeBot):
    def __init__(self, rich_error: Exception | None = None):
        super().__init__(rich_error)
        self.photos: list[dict] = []

    async def send_photo(self, **kwargs):
        self.photos.append(kwargs)


def test_captcha_futures_are_independent_between_users(tmp_path: Path) -> None:
    """Stage 4\7: two users can await CAPTCHA futures independently."""
    telegram, _, _, bot = service(tmp_path)
    bot = FakeBotWithPhotos()
    telegram.bot = bot

    async def scenario():
        task_a = asyncio.create_task(
            telegram.request_captcha(
                Path("/tmp/captcha_a.png"), "Vacancy A", 60, telegram_user_id=42
            )
        )
        task_b = asyncio.create_task(
            telegram.request_captcha(
                Path("/tmp/captcha_b.png"), "Vacancy B", 60, telegram_user_id=43
            )
        )
        await asyncio.sleep(0)
        # Both futures exist and are distinct
        assert set(telegram._captcha_futures) == {42, 43}
        assert telegram._captcha_futures[42] is not telegram._captcha_futures[43]

        # Answer user 43 first — only their future resolves
        telegram._captcha_futures[43].set_result("answer-b")
        result_b = await task_b
        assert result_b == "answer-b"
        assert not task_a.done()

        # Answer user 42
        telegram._captcha_futures[42].set_result("answer-a")
        result_a = await task_a
        assert result_a == "answer-a"

        # Both entries cleaned up
        assert telegram._captcha_futures == {}

    asyncio.run(scenario())


def test_captcha_cancel_one_user_does_not_affect_other(tmp_path: Path) -> None:
    """Stage 4\7: /cancel resolves only the caller's CAPTCHA future."""
    telegram, _, _, bot = service(tmp_path)
    bot = FakeBotWithPhotos()
    telegram.bot = bot

    async def scenario():
        task_a = asyncio.create_task(
            telegram.request_captcha(
                Path("/tmp/captcha_a.png"), "Vacancy A", 60, telegram_user_id=42
            )
        )
        task_b = asyncio.create_task(
            telegram.request_captcha(
                Path("/tmp/captcha_b.png"), "Vacancy B", 60, telegram_user_id=43
            )
        )
        await asyncio.sleep(0)

        # User 42 sends /cancel
        reply = telegram.command("cancel", user_id=42)
        assert reply == "Current input request cancelled."

        # User 42's future resolved with None; user 43's still pending
        result_a = await task_a
        assert result_a is None
        assert not task_b.done()
        assert not telegram._captcha_futures[43].done()

        # User 43 still able to answer
        telegram._captcha_futures[43].set_result("answer-b")
        assert await task_b == "answer-b"
        assert telegram._captcha_futures == {}

    asyncio.run(scenario())


def test_captcha_text_answer_goes_to_correct_user(tmp_path: Path) -> None:
    """Stage 4\7: a text reply resolves only the sender's CAPTCHA future."""
    telegram, database, _, bot = service(tmp_path)
    bot = FakeBotWithPhotos()
    telegram.bot = bot

    database.create_user(
        telegram_user_id=43,
        username="testuser43",
        now=NOW,
    )

    async def scenario():
        task_a = asyncio.create_task(
            telegram.request_captcha(
                Path("/tmp/captcha_a.png"), "Vacancy A", 60, telegram_user_id=42
            )
        )
        task_b = asyncio.create_task(
            telegram.request_captcha(
                Path("/tmp/captcha_b.png"), "Vacancy B", 60, telegram_user_id=43
            )
        )
        await asyncio.sleep(0)

        # User 43 replies with the CAPTCHA text
        user_43 = FakeUser(user_id=43)
        message = FakeMessage(text="captcha-text-43", from_user=user_43)
        await telegram._text_handler(message)
        assert message.answer_text == "CAPTCHA input received."

        # Only user 43's future resolved
        assert await task_b == "captcha-text-43"
        assert not task_a.done()
        assert not telegram._captcha_futures[42].done()

        telegram._captcha_futures[42].set_result("answer-a")
        assert await task_a == "answer-a"
        assert telegram._captcha_futures == {}

    asyncio.run(scenario())


def test_captcha_cleanup_removes_only_own_future(tmp_path: Path) -> None:
    """Stage 4\7: cleanup of a finished request never drops a newer future."""
    telegram, _, _, bot = service(tmp_path)
    bot = FakeBotWithPhotos()
    telegram.bot = bot

    async def scenario():
        # First request for user 42 finishes (timeout)
        task_a = asyncio.create_task(
            telegram.request_captcha(
                Path("/tmp/captcha_a.png"), "Vacancy A", 1, telegram_user_id=42
            )
        )
        # While it is still registered, a second request for the same user starts
        await asyncio.sleep(0)
        future_first = telegram._captcha_futures[42]
        task_a2 = asyncio.create_task(
            telegram.request_captcha(
                Path("/tmp/captcha_a2.png"), "Vacancy A2", 60, telegram_user_id=42
            )
        )
        await asyncio.sleep(0)
        future_second = telegram._captcha_futures[42]
        assert future_second is not future_first

        # Let the first request time out
        result = await task_a
        assert result is None

        # The newer future must survive cleanup of the old one
        assert telegram._captcha_futures[42] is future_second

        future_second.set_result("answer-a2")
        assert await task_a2 == "answer-a2"
        assert telegram._captcha_futures == {}

    asyncio.run(scenario())


def test_captcha_send_photo_error_cleans_up_future(tmp_path: Path) -> None:
    """Stage 4\7: send_photo exception cleans up user's future and leaves others intact."""
    telegram, _, _, _ = service(tmp_path)

    class ErrorBot(FakeBot):
        async def send_photo(self, **kwargs):
            raise TelegramNetworkError(method=None, message="network error during send_photo")

    telegram.bot = ErrorBot()

    async def scenario():
        other_future = asyncio.get_running_loop().create_future()
        telegram._captcha_futures[43] = other_future

        with pytest.raises(TelegramNetworkError):
            await telegram.request_captcha(
                Path("/tmp/captcha.png"), "Vacancy Error", 60, telegram_user_id=42
            )

        assert 42 not in telegram._captcha_futures
        assert telegram._captcha_futures.get(43) is other_future

        other_future.set_result("ok")

    asyncio.run(scenario())


# ============================================================================
# Phase 5.6 - ownership-checked /logout and verified /resume
# ============================================================================


class FakeSessionClient:
    """Deterministic HHClient surface for Telegram command tests."""

    def __init__(
        self,
        telegram_user_id: int,
        *,
        state: LoginState = LoginState.AUTHORIZED,
        logout_result: LogoutResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.telegram_user_id = telegram_user_id
        self.state = state
        self.error = error
        self.logout_result = logout_result or LogoutResult(
            True, LoginState.NOT_AUTHORIZED, "confirmed"
        )
        self.check_calls = 0
        self.logout_calls = 0
        self.session_lock = asyncio.Lock()

    async def check_session(self) -> LoginState:
        self.check_calls += 1
        if self.error is not None:
            raise self.error
        return self.state

    async def logout(self) -> LogoutResult:
        self.logout_calls += 1
        if self.error is not None:
            raise self.error
        return self.logout_result


def _phase56_service(tmp_path: Path):
    settings = load_settings(
        profile_path=write_profile(tmp_path),
        environ={**VALID_ENV, "APP_MODE": "dry_run"},
    )
    settings = replace(settings, database_path=tmp_path / "agent.db")
    database = Database(settings.database_path)
    database.init()
    for user_id in (42, 43):
        database.create_user(telegram_user_id=user_id, now=NOW)
        database.get_or_create_hh_account(telegram_user_id=user_id, now=NOW)

    contexts = {}
    clients = {}
    for user_id in (42, 43):
        context = create_runtime_context(
            user_id, settings, database, load_user_settings(settings)
        )
        context.auth_state = LoginState.AUTHORIZED.value
        context.auth_paused = False
        contexts[user_id] = context
        clients[user_id] = FakeSessionClient(user_id)

    telegram = TelegramService(
        settings,
        database,
        FakeApprovalService(),
        contexts=contexts,
        bot=FakeBot(),
        now_factory=lambda: NOW,
        hh_clients=clients,
    )
    return telegram, contexts, clients


def test_logout_confirmed_pauses_only_the_callers_runtime(tmp_path: Path) -> None:
    telegram, contexts, clients = _phase56_service(tmp_path)

    reply = asyncio.run(telegram.command_async("logout", 42))

    assert "confirmed" in reply.casefold()
    assert clients[42].logout_calls == 1
    assert clients[43].logout_calls == 0
    assert contexts[42].auth_state == LoginState.NOT_AUTHORIZED.value
    assert contexts[42].auth_paused is True
    assert contexts[42].auth_reason == "logged_out"
    assert contexts[43].auth_state == LoginState.AUTHORIZED.value
    assert contexts[43].auth_paused is False


def test_unconfirmed_logout_does_not_claim_the_session_closed(tmp_path: Path) -> None:
    telegram, contexts, clients = _phase56_service(tmp_path)
    clients[42].logout_result = LogoutResult(
        False, LoginState.AUTHORIZED, "unconfirmed"
    )
    contexts[42].auth_paused = True
    contexts[42].auth_reason = "existing_pause"

    reply = asyncio.run(telegram.command_async("logout", 42))

    assert "not confirmed" in reply.casefold()
    assert contexts[42].auth_state == LoginState.AUTHORIZED.value
    assert contexts[42].auth_paused is True
    assert contexts[42].auth_reason == "existing_pause"


def test_failed_logout_check_fails_runtime_closed(tmp_path: Path) -> None:
    """An unconfirmed logout must not leave search enabled."""
    telegram, contexts, clients = _phase56_service(tmp_path)
    clients[42].logout_result = LogoutResult(
        False, LoginState.CHECK_FAILED, "invalidate_failed"
    )

    reply = asyncio.run(telegram.command_async("logout", 42))

    assert "could not be confirmed" in reply.casefold()
    assert contexts[42].auth_state == LoginState.CHECK_FAILED.value
    assert contexts[42].auth_paused is True
    assert contexts[42].auth_reason == "session_check_unconfirmed"
    assert contexts[43].auth_paused is False


def test_ownership_failed_logout_fails_runtime_closed(tmp_path: Path) -> None:
    """A rejected logout identity must not leave stale runtime access."""
    telegram, contexts, clients = _phase56_service(tmp_path)
    clients[42].logout_result = LogoutResult(
        False, LoginState.CHECK_FAILED, "ownership"
    )

    reply = asyncio.run(telegram.command_async("logout", 42))

    assert "ownership" in reply.casefold()
    assert contexts[42].auth_state == LoginState.CHECK_FAILED.value
    assert contexts[42].auth_paused is True
    assert contexts[42].auth_reason == "session_check_unconfirmed"
    assert contexts[43].auth_paused is False


def test_resume_only_clears_auth_pause_after_positive_check(tmp_path: Path) -> None:
    telegram, contexts, clients = _phase56_service(tmp_path)
    context = contexts[42]
    context.paused = True
    context.auth_paused = True
    context.auth_reason = "logged_out"
    context.circuit_reason = "page_structure_changed"
    context.consecutive_search_errors = 3

    reply = asyncio.run(telegram.command_async("resume", 42))

    assert reply == "Agent resumed."
    assert clients[42].check_calls == 1
    assert context.auth_state == LoginState.AUTHORIZED.value
    assert context.auth_paused is False
    assert context.auth_reason == ""
    assert context.paused is False
    assert context.circuit_reason == ""
    assert context.consecutive_search_errors == 0
    assert context.wake_event.is_set()


def _auth_snapshot(context):
    return (
        context.auth_state,
        context.auth_paused,
        context.auth_reason,
        context.paused,
        context.circuit_reason,
        context.consecutive_search_errors,
        context.next_run_at,
        context.wake_event.is_set(),
    )


def test_resume_without_hh_client_does_not_clear_pause(tmp_path: Path) -> None:
    telegram, contexts, clients = _phase56_service(tmp_path)
    context = contexts[42]
    telegram._hh_clients.clear()
    context.paused = True
    context.auth_paused = True
    context.auth_reason = "logged_out"
    before = _auth_snapshot(context)

    reply = asyncio.run(telegram.command_async("resume", 42))

    assert "not available" in reply.casefold()
    assert _auth_snapshot(context) == before


def test_sync_resume_without_hh_client_does_not_clear_pause(
    tmp_path: Path,
) -> None:
    telegram, contexts, clients = _phase56_service(tmp_path)
    context = contexts[42]
    telegram._hh_clients.clear()
    context.paused = True
    context.auth_paused = True
    context.auth_reason = "logged_out"
    before = _auth_snapshot(context)

    reply = telegram.command("resume", 42)

    assert "not available" in reply.casefold()
    assert _auth_snapshot(context) == before


@pytest.mark.parametrize("entry_point", ["async", "sync"])
def test_missing_hh_client_leaves_all_runtime_control_fields_untouched(
    tmp_path: Path, entry_point: str
) -> None:
    telegram, contexts, clients = _phase56_service(tmp_path)
    context = contexts[42]
    telegram._hh_clients.clear()
    context.auth_state = LoginState.NOT_AUTHORIZED.value
    context.auth_paused = True
    context.auth_reason = "logged_out"
    context.paused = True
    context.circuit_reason = "page_structure_changed"
    context.consecutive_search_errors = 3
    context.next_run_at = NOW + timedelta(minutes=30)
    before = _auth_snapshot(context)

    if entry_point == "async":
        reply = asyncio.run(telegram.command_async("resume", 42))
    else:
        reply = telegram.command("resume", 42)

    assert "not available" in reply.casefold()
    assert context.auth_state == before[0]
    assert context.auth_paused is before[1]
    assert context.paused is before[3]
    assert context.wake_event.is_set() is before[7]
    assert _auth_snapshot(context) == before


@pytest.mark.parametrize("entry_point", ["async", "sync"])
def test_wrong_client_identity_cannot_resume(
    tmp_path: Path, entry_point: str
) -> None:
    telegram, contexts, clients = _phase56_service(tmp_path)
    context = contexts[42]
    registered = FakeSessionClient(43)
    telegram._hh_clients[42] = registered  # type: ignore[assignment]
    context.auth_paused = True
    context.auth_reason = "logged_out"
    context.paused = True
    before = _auth_snapshot(context)

    if entry_point == "async":
        reply = asyncio.run(telegram.command_async("resume", 42))
    else:
        reply = telegram.command("resume", 42)

    assert "not available" in reply.casefold()
    assert registered.check_calls == 0
    assert _auth_snapshot(context) == before
    assert contexts[43].auth_paused is False


def test_positive_authorized_check_still_allows_resume(tmp_path: Path) -> None:
    telegram, contexts, clients = _phase56_service(tmp_path)
    context = contexts[42]
    clients[42].state = LoginState.AUTHORIZED
    context.paused = True
    context.auth_paused = True
    context.auth_reason = "logged_out"

    reply = asyncio.run(telegram.command_async("resume", 42))

    assert reply == "Agent resumed."
    assert clients[42].check_calls == 1
    assert context.auth_state == LoginState.AUTHORIZED.value
    assert context.auth_paused is False
    assert context.auth_reason == ""
    assert context.paused is False
    assert context.wake_event.is_set()
    assert contexts[43].auth_paused is False


@pytest.mark.parametrize(
    ("state", "expected_reason"),
    [
        (LoginState.NOT_AUTHORIZED, "not_authorized"),
        (LoginState.UNKNOWN, "session_check_unconfirmed"),
        (LoginState.CHECK_FAILED, "session_check_unconfirmed"),
    ],
)
def test_resume_keeps_pause_without_positive_session_check(
    tmp_path: Path, state: LoginState, expected_reason: str
) -> None:
    telegram, contexts, clients = _phase56_service(tmp_path)
    context = contexts[42]
    context.paused = True
    context.auth_paused = True
    context.auth_reason = "logged_out"
    context.circuit_reason = "page_structure_changed"
    clients[42].state = state

    reply = asyncio.run(telegram.command_async("resume", 42))

    assert "could not be confirmed" in reply.casefold() or (
        "authorization is required" in reply.casefold()
    )
    assert context.auth_state == state.value
    assert context.auth_paused is True
    assert context.auth_reason == expected_reason
    assert context.paused is True
    assert context.circuit_reason == "page_structure_changed"
    assert not context.wake_event.is_set()
    assert contexts[43].auth_paused is False


def test_foreign_client_and_context_cannot_be_controlled(tmp_path: Path) -> None:
    telegram, contexts, clients = _phase56_service(tmp_path)
    own_context, other_context = contexts[42], contexts[43]
    own_client, other_client = clients[42], clients[43]
    own_context.auth_paused = True

    telegram._contexts[42] = other_context
    assert asyncio.run(telegram.command_async("logout", 42)) == (
        "Agent runtime is not running for your account."
    )
    assert other_client.logout_calls == 0
    telegram._contexts[42] = own_context

    telegram._hh_clients[42] = other_client
    assert asyncio.run(telegram.command_async("logout", 42)) == (
        "HH session manager is not available."
    )
    assert other_client.logout_calls == 0
    assert own_context.auth_paused is True

    assert asyncio.run(telegram.command_async("resume", 42)) == (
        "HH session manager is not available."
    )
    assert other_client.check_calls == 0
    assert own_context.auth_paused is True


@pytest.mark.parametrize("command", ["logout", "resume"])
def test_session_check_error_is_contained_by_command_handler(
    tmp_path: Path, command: str
) -> None:
    telegram, contexts, clients = _phase56_service(tmp_path)
    clients[42].error = RuntimeError("token=SECRET-DO-NOT-SHOW")
    contexts[42].auth_paused = True
    contexts[42].auth_reason = "logged_out"
    message = FakeMessage(f"/{command}", FakeUser(42, "testuser42"))

    asyncio.run(telegram._command_handler(message))

    assert message.answer_text is not None
    assert "SECRET" not in message.answer_text
    assert "could not be confirmed" in message.answer_text.casefold()
    assert contexts[42].auth_paused is True
    assert contexts[42].auth_reason == "logged_out"