"""Tests for updater module.

Stage 4\3: verify update notification deduplication works with explicit telegram_user_id.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from config import load_settings
from database import Database
from runtime_context import create_runtime_context
from tg_bot import TelegramService
from updater import ReleaseInfo


class FakeBot:
    def __init__(self):
        self.messages: list[dict] = []

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)


class FakeApprovalService:
    async def approve_and_apply(self, job_id: str, user_id: int):
        raise AssertionError("approval must not be called")

    def skip(self, job_id: str, user_id: int):
        raise AssertionError("skip must not be called")


def _telegram_service(tmp_path: Path):
    """Helper: create TelegramService with fake bot."""
    from dataclasses import replace

    settings = load_settings(
        profile_path=None,
        environ={
            "TG_BOT_TOKEN": "fake-token",
            "APP_MODE": "dry_run",
            "ENABLE_REAL_APPLY": "false",
            "BROWSER_HEADLESS": "false",
        },
    )
    settings = replace(settings, database_path=tmp_path / "agent.db")
    database = Database(settings.database_path)
    database.init()

    context = create_runtime_context(42, settings, database)
    bot = FakeBot()
    telegram = TelegramService(
        settings,
        database,
        FakeApprovalService(),
        contexts={42: context},
        bot=bot,
    )
    return telegram, bot


def test_telegram_check_updates_sends_notification_once(tmp_path: Path) -> None:
    """check_updates() sends update notification once per release tag.

    The deduplication logic lives in check_updates(), not notify_update_available().
    _notified_release_tag is checked and set inside check_updates() before
    calling notify_update_available(). Two calls to check_updates() with the
    same release tag should result in exactly one notification.
    """
    telegram, bot = _telegram_service(tmp_path)

    fake_release = ReleaseInfo(
        tag_name="v2.0.0",
        html_url="https://example.com/release",
        name="v2.0.0",
        published_at="2026-09-01T00:00:00Z",
        body="Update notes",
    )

    # Monkeypatch check_github_release to return our fake release
    with patch("tg_bot.check_github_release", return_value=fake_release):
        # First check_updates call should notify
        result1 = asyncio.run(
            telegram.check_updates(notify=True, telegram_user_id=42)
        )
        assert result1 is not None
        assert result1.tag_name == "v2.0.0"
        assert len(bot.messages) == 1
        assert bot.messages[0]["chat_id"] == 42
        assert "v2.0.0" in bot.messages[0]["text"]

        # Second check_updates call with same tag should NOT notify again
        result2 = asyncio.run(
            telegram.check_updates(notify=True, telegram_user_id=42)
        )
        assert result2 is not None
        assert result2.tag_name == "v2.0.0"
        assert len(bot.messages) == 1  # Still only one message


def test_check_updates_without_telegram_user_id_does_not_notify(tmp_path: Path) -> None:
    """check_updates() without telegram_user_id does not send notification."""
    telegram, bot = _telegram_service(tmp_path)

    fake_release = ReleaseInfo(
        tag_name="v2.0.0",
        html_url="https://example.com/release",
        name="v2.0.0",
        published_at="2026-09-01T00:00:00Z",
        body="Update notes",
    )

    with patch("tg_bot.check_github_release", return_value=fake_release):
        result = asyncio.run(
            telegram.check_updates(notify=True, telegram_user_id=None)
        )
        assert result is not None
        assert len(bot.messages) == 0  # No notification sent