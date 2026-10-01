from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from config import load_user_settings

if TYPE_CHECKING:
    from config import Settings, UserSettings
    from database import Database


@dataclass
class UserRuntimeContext:
    """Runtime context for a single Telegram user.

    Stage 4\4: this object is the sole owner of per-user runtime control
    state (pause/resume, circuit breaker, wake signalling). There is no
    shared/global AgentControl anymore — every field below belongs to
    exactly one Telegram user.

    Stage 4\6: the context also owns the user's isolated configuration
    snapshot (``user_settings``).  Exactly ONE :class:`config.UserSettings`
    instance is shared by the context and every user-scoped component
    (HHClient, ApprovalGuard, ApprovalService, process_vacancy,
    run_search_cycle, TelegramService/status), so changing user 42's
    settings can never affect user 43.
    """

    telegram_user_id: int
    settings: Settings
    database: Database
    # Stage 4\6: the single per-user configuration instance shared by all
    # user-scoped components.  When not passed explicitly the factory
    # materialises a fresh snapshot from ``settings.user_defaults`` —
    # never a shared mutable object.
    user_settings: UserSettings

    # Per-user runtime control state (replaces the old shared AgentControl)
    paused: bool = False
    circuit_reason: str = ""
    consecutive_search_errors: int = 0
    next_run_at: datetime | None = None
    wake_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    # Diagnostics
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())


def create_runtime_context(
    telegram_user_id: int,
    settings: Settings,
    database: Database,
    user_settings: UserSettings | None = None,
) -> UserRuntimeContext:
    """Factory: create a fresh UserRuntimeContext for the given user.

    Stage 4\6: ``user_settings`` may be the already-materialised per-user
    configuration instance (production path — stored by identity) or
    ``None`` (legacy callers), in which case a fresh snapshot is created
    from the immutable global defaults.  Two contexts therefore never
    share a ``UserSettings`` instance::

        ctx_42 = create_runtime_context(42, settings, database)
        ctx_43 = create_runtime_context(43, settings, database)
        assert ctx_42.user_settings is not ctx_43.user_settings
    """
    if user_settings is None:
        user_settings = load_user_settings(settings)
    return UserRuntimeContext(
        telegram_user_id=telegram_user_id,
        settings=settings,
        database=database,
        user_settings=user_settings,
    )