from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Union

from config import Settings, UserSettings, load_user_settings
from database import ClaimResult, Database


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApplicationPermission:
    job_id: str
    permit: str
    telegram_user_id: int


@dataclass(frozen=True)
class ApprovalResult:
    ok: bool
    message: str


class PhysicalApplicationSender(Protocol):
    async def submit_application(self, permission: ApplicationPermission) -> bool: ...


def _as_user_settings(source: Union[Settings, UserSettings]) -> UserSettings:
    """Normalise a caller-supplied configuration to a UserSettings snapshot.

    Stage 4\6: legacy callers pass the global :class:`Settings`; we
    materialise a fresh per-user snapshot from its immutable defaults.
    Callers that already pass a :class:`UserSettings` get it back
    unchanged (identity preserved — no copy, no shared mutable state).
    """
    if isinstance(source, UserSettings):
        return source
    return load_user_settings(source)


class ApprovalGuard:
    """Guards the physical application step for one Telegram user.

    Stage 4\6: the guard operates on the *caller's* :class:`UserSettings`
    (app mode, real-apply flag, daily limit), never on a shared global
    mutable configuration.  The constructor accepts either a
    :class:`UserSettings` (production path, stored by identity) or the
    legacy global :class:`Settings` (normalised to a fresh snapshot).
    """

    def __init__(
        self,
        settings: Union[Settings, UserSettings],
        database: Database,
        now_factory: Callable[[], datetime] | None = None,
    ):
        self.user_settings = _as_user_settings(settings)
        self.database = database
        self.now_factory = now_factory or (lambda: datetime.now().astimezone())

    def claim(self, permission: ApplicationPermission) -> ClaimResult:
        return self.database.claim_application(
            job_id=permission.job_id,
            permit=permission.permit,
            telegram_user_id=permission.telegram_user_id,
            expected_user_id=permission.telegram_user_id,
            app_mode=self.user_settings.app_mode,
            enable_real_apply=self.user_settings.enable_real_apply,
            daily_limit=self.user_settings.max_applications_per_day,
            now=self.now_factory(),
        )


class ApprovalService:
    """User-facing approval flow backed by per-user settings.

    Stage 4\6: mode/limits come from the user's own immutable
    ``UserSettings`` instance, so user 42's approval flow can never be
    influenced by user 43's configuration.  Accepts either
    :class:`UserSettings` (identity preserved) or legacy
    :class:`Settings` (normalised to a fresh snapshot).
    """

    def __init__(
        self,
        settings: Union[Settings, UserSettings],
        database: Database,
        sender: PhysicalApplicationSender,
        now_factory: Callable[[], datetime] | None = None,
    ):
        self.user_settings = _as_user_settings(settings)
        self.database = database
        self.sender = sender
        self.now_factory = now_factory or (lambda: datetime.now().astimezone())

    async def approve_and_apply(
        self, job_id: str, telegram_user_id: int
    ) -> ApprovalResult:
        if self.user_settings.app_mode != "approval":
            logger.warning("application_blocked job_id=%s reason=app_mode", job_id)
            return ApprovalResult(False, "Real applications are disabled in dry-run")
        if not self.user_settings.enable_real_apply:
            logger.warning("application_blocked job_id=%s reason=real_apply_disabled", job_id)
            return ApprovalResult(False, "Real applications are disabled")
        permit = self.database.approve(
            job_id,
            telegram_user_id,
            telegram_user_id,
            self.now_factory(),
            self.user_settings.approval_ttl_minutes,
        )
        if permit is None:
            logger.warning("application_blocked job_id=%s reason=approval_invalid", job_id)
            return ApprovalResult(False, "Approval is missing, expired, or already used")
        logger.info("approval_received job_id=%s", job_id)
        permission = ApplicationPermission(job_id, permit, telegram_user_id)
        sent = await self.sender.submit_application(permission)
        return ApprovalResult(sent, "Application sent" if sent else "Application failed")

    def skip(self, job_id: str, telegram_user_id: int) -> ApprovalResult:
        skipped = self.database.skip(
            job_id, telegram_user_id, telegram_user_id
        )
        return ApprovalResult(skipped, "Vacancy skipped" if skipped else "Vacancy is not pending")