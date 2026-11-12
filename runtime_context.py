from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from config import load_user_settings

if TYPE_CHECKING:
    from config import Settings, UserSettings
    from database import Database, HHAccount


class HHAccountOwnershipError(RuntimeError, ValueError):
    """An HH account does not belong to the expected Telegram user."""


def verify_hh_account_ownership(
    database: Database,
    telegram_user_id: int,
    hh_account: HHAccount | None,
) -> HHAccount:
    """Return the verified database record for an HH account.

    Phase 5.4: ownership is decided by the database, not by fields copied
    into a Python object.  The account id and Telegram user id must match
    the same ``hh_accounts`` row, and the object's browser profile path
    must equal the path stored on that row.
    """
    if hh_account is None:
        raise HHAccountOwnershipError(
            "HH account is required for an authenticated Telegram user"
        )

    if hh_account.telegram_user_id != telegram_user_id:
        raise HHAccountOwnershipError(
            "HH account does not belong to this Telegram user"
        )

    verified = database.get_hh_account_by_id(
        hh_account.id,
        telegram_user_id,
    )
    if verified is None:
        raise HHAccountOwnershipError(
            "HH account does not exist for this Telegram user"
        )

    if verified.telegram_user_id != telegram_user_id:
        raise HHAccountOwnershipError(
            "HH account does not belong to this Telegram user"
        )

    if verified.browser_profile_dir != hh_account.browser_profile_dir:
        raise HHAccountOwnershipError(
            "HH account browser profile does not match the stored "
            "account record"
        )

    return verified


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

    Phase 5.4: an HH account attached to this context is verified against
    the database by BOTH its account id and this context's Telegram user
    id.  The identity fields cannot be reassigned after construction, so
    ownership cannot be bypassed by mutating the context later.
    """

    telegram_user_id: int
    settings: Settings
    database: Database
    # Stage 4\6: the single per-user configuration instance shared by all
    # user-scoped components.  When not passed explicitly the factory
    # materialises a fresh snapshot from ``settings.user_defaults`` —
    # never a shared mutable object.
    user_settings: UserSettings
    hh_account: HHAccount | None = None

    # Per-user runtime control state (replaces the old shared AgentControl)
    paused: bool = False
    circuit_reason: str = ""
    consecutive_search_errors: int = 0
    next_run_at: datetime | None = None
    wake_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    # Diagnostics
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    # Phase 5.4: ownership identity is immutable after __post_init__.
    _identity_locked: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.hh_account is not None:
            verified = verify_hh_account_ownership(
                self.database,
                self.telegram_user_id,
                self.hh_account,
            )
            # Store the database record, not a caller-supplied copy.
            object.__setattr__(self, "hh_account", verified)
        object.__setattr__(self, "_identity_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"telegram_user_id", "hh_account"} and getattr(
            self, "_identity_locked", False
        ):
            raise AttributeError(
                f"{name} cannot be changed after UserRuntimeContext construction"
            )
        object.__setattr__(self, name, value)


def create_runtime_context(
    telegram_user_id: int,
    settings: Settings,
    database: Database,
    user_settings: UserSettings | None = None,
    hh_account: HHAccount | None = None,
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

    Phase 5.4: when ``hh_account`` is supplied it is verified against the
    database before the context is constructed.  When omitted, the
    factory loads the Telegram user's own account and verifies it the
    same way (``None`` is allowed for legacy contexts without an account).
    """
    if user_settings is None:
        user_settings = load_user_settings(settings)
    if hh_account is None:
        hh_account = database.get_hh_account_model(telegram_user_id)
    return UserRuntimeContext(
        telegram_user_id=telegram_user_id,
        settings=settings,
        database=database,
        user_settings=user_settings,
        hh_account=hh_account,
    )
