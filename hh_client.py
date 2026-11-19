from __future__ import annotations

import asyncio
import inspect
import logging
import random
import re
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse
from uuid import uuid4

from approval import ApplicationPermission, ApprovalGuard
from config import Settings, UserSettings, load_user_settings
from database import Database, HHAccount, HHAccountStatus, VacancyStatus
from runtime_context import HHAccountOwnershipError, verify_hh_account_ownership
from vacancy_filter import is_relevant_logistics_title


HH_BASE_URL = "https://hh.kz"

# Phase 5.5/5.6: persisted HH account statuses used by session verification.
HH_ACCOUNT_STATUS_AUTHORIZED = HHAccountStatus.AUTHORIZED.value
HH_ACCOUNT_STATUS_LOGIN_REQUIRED = HHAccountStatus.LOGIN_REQUIRED.value
HH_ACCOUNT_STATUS_UNKNOWN = HHAccountStatus.UNKNOWN.value
HH_ACCOUNT_STATUS_CHECK_FAILED = HHAccountStatus.CHECK_FAILED.value
HH_ACCOUNT_STATUS_SESSION_EXPIRED = HHAccountStatus.SESSION_EXPIRED.value
HH_ACCOUNT_STATUS_LOGGED_OUT = HHAccountStatus.LOGGED_OUT.value

# These persisted states all prove that the account was positively
# authorised at some point.  They replace last_login_at as the memory used
# to distinguish SESSION_EXPIRED from a first-time LOGIN_REQUIRED result.
_PRIOR_AUTHORIZATION_STATUSES = frozenset(
    {
        HH_ACCOUNT_STATUS_AUTHORIZED,
        HH_ACCOUNT_STATUS_SESSION_EXPIRED,
        HH_ACCOUNT_STATUS_LOGGED_OUT,
    }
)

# Phase 5.5: authorisation is confirmed only by POSITIVE evidence.  These
# are the page signals used to recognise an authenticated applicant area.
# Several independent markers are checked so a minor HH.kz markup change
# does not silently turn a real session into a false negative.
LOGGED_IN_MARKERS = (
    '[data-qa="mainmenu_applicantProfile"]',
    '[data-qa="mainmenu_profile"]',
    '[data-qa="mainmenu_resumes"]',
    '[data-qa="mainmenu_negotiations"]',
    '[data-qa="resume-title"]',
    '[data-qa*="resume-title"]',
    '[data-qa="resume-list"]',
    'a[href^="/applicant/resumes"]',
    'h1:has-text("Мои резюме")',
)

LOGIN_INDICATORS = (
    'input[type="tel"]',
    'input[data-qa*="login"]',
    'a[data-qa="mainmenu_applicantAccess"]',
    '[data-qa="account-login"]',
    '[data-qa*="login-submit"]',
    'a:has-text("Войти")',
    'button:has-text("Войти")',
)

# Safe, secret-free error texts persisted to hh_accounts.last_error.
SAFE_LOGIN_ERROR_HEADLESS = (
    "HH.kz authorization is required and browser_headless is enabled"
)
SAFE_LOGIN_ERROR_UNVERIFIED = (
    "HH.kz authorization could not be verified from the page state"
)
SAFE_LOGIN_ERROR_MANUAL = (
    "HH.kz authorization was not confirmed after manual login"
)
SAFE_LOGIN_ERROR_NAVIGATION = (
    "HH.kz login page could not be loaded"
)
SAFE_LOGIN_ERROR_CHECK_FAILED = (
    "HH.kz session check failed due to a technical error"
)
SAFE_LOGIN_ERROR_SESSION_EXPIRED = (
    "HH.kz session expired: authorization is required again"
)
SAFE_LOGIN_ERROR_LOGGED_OUT = (
    "HH.kz session was closed by the user"
)

logger = logging.getLogger(__name__)

LOGISTICS_TITLE_KEYWORDS = (
    "логист",
    "перевоз",
    "транспорт",
    "экспедитор",
    "диспетчер",
    "вэд",
    "тамож",
    "supply chain",
    "supply-chain",
    "freight",
    "logistics",
    "transport",
)


def _is_logistics_vacancy(title: str) -> bool:
    normalized = title.casefold().replace("ё", "е")

    return any(
        keyword in normalized
        for keyword in LOGISTICS_TITLE_KEYWORDS
    )


def _is_logistics_query(query: str) -> bool:
    normalized = query.casefold().replace("ё", "е")

    return any(
        keyword in normalized
        for keyword in LOGISTICS_TITLE_KEYWORDS
    )


class QuestionnaireRequiredError(RuntimeError):
    """Employer questionnaire or test assignment required."""
    pass


class LoginState(str, Enum):
    """Outcome of a single HH.kz authorisation check.

    Phase 5.5/5.6: ``UNKNOWN`` and ``CHECK_FAILED`` are first-class
    outcomes.  A technical failure is never evidence of a logout, and an
    inconclusive page is never treated as an authenticated session.
    """

    AUTHORIZED = "authorized"
    NOT_AUTHORIZED = "not_authorized"
    UNKNOWN = "unknown"
    CHECK_FAILED = "check_failed"


@dataclass(frozen=True)
class LogoutResult:
    """Secret-free result of an HH account logout attempt."""

    ok: bool
    state: LoginState
    reason: str


def _utc_iso(value: datetime) -> str:
    """Return an ISO-8601 UTC timestamp for persisted account state."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _page_url(page: Any) -> str:
    """Read the current page URL without trusting a specific page class."""
    try:
        value = getattr(page, "url", "")
        if callable(value):
            value = value()
        return str(value or "")
    except Exception:
        return ""


def _is_hh_origin(url: str) -> bool:
    """True only for https://hh.kz[:443] or https://www.hh.kz[:443].

    Any other scheme, host, port or embedded credentials are rejected.
    Malformed URLs return False instead of raising.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme.casefold() != "https":
            return False
        if parsed.hostname not in {"hh.kz", "www.hh.kz"}:
            return False
        if parsed.username is not None or parsed.password is not None:
            return False
        port = parsed.port
        return port is None or port == 443
    except (AttributeError, TypeError, ValueError):
        return False


def _is_login_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.casefold()
    return any(
        fragment in path
        for fragment in ("/account/login", "/logon", "/auth/login", "/auth/")
    )


def _is_hh_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").casefold()
    return hostname == "hh.kz" or hostname.endswith(".hh.kz") or (
        hostname == "hh.ru" or hostname.endswith(".hh.ru")
    )


def _is_applicant_url(url: str) -> bool:
    """True only for an authenticated-area URL on an HH host."""
    if not _is_hh_url(url) or _is_login_url(url):
        return False
    return urlparse(url).path.casefold().startswith("/applicant")


async def _any_visible(page: Any, selectors: tuple[str, ...]) -> bool:
    """Return True when at least one selector is currently visible."""
    for selector in selectors:
        try:
            if await page.locator(selector).first.is_visible():
                return True
        except Exception as exc:
            logger.debug(
                "hh_login_selector_check_failed selector=%s error=%s",
                selector,
                type(exc).__name__,
            )
    return False


async def _console_manual_login_prompt(message: str) -> None:
    await asyncio.to_thread(input, message)


class PageState(str, Enum):
    VACANCY_LOADED = "vacancy_loaded"
    CAPTCHA_DETECTED = "captcha_detected"
    ACCESS_DENIED = "access_denied"
    VACANCY_REMOVED = "vacancy_removed"
    PAGE_STRUCTURE_CHANGED = "page_structure_changed"
    NETWORK_ERROR = "network_error"


@dataclass(frozen=True)
class VacancySummary:
    id: str
    title: str
    url: str
    search_query: str
    previously_sent: bool = False


@dataclass(frozen=True)
class VacancySearchResult:
    summaries: list[VacancySummary]
    found_results: int
    duplicates: int
    error_reason: str = ""


@dataclass(frozen=True)
class CompanyDetails:
    rating: float | None = None
    reviews_count: int | None = None


@dataclass(frozen=True)
class VacancyDetails:
    summary: VacancySummary
    state: PageState
    company: str = ""
    description: str = ""
    error: str = ""
    company_url: str = ""
    salary: str = ""
    schedule: str = ""
    location: str = ""
    experience: str = ""


CaptchaSolver = Callable[[Path, str, int], Awaitable[str | None]]


def _vacancy_id(url: str) -> str | None:
    match = re.search(r"(?:^|/)vacancy/(\d+)(?:/|$)", urlparse(url).path)
    return match.group(1) if match else None


async def _visible_text(page: Any, selector: str) -> str:
    locator = page.locator(selector)
    return (await locator.inner_text()).strip() if await locator.is_visible() else ""


def _company_url(vacancy_url: str, href: str | None) -> str:
    if not href:
        return ""

    parsed = urlparse(urljoin(vacancy_url, href))
    hostname = (parsed.hostname or "").lower()

    if (
        parsed.scheme != "https"
        or not (
            hostname == "hh.kz"
            or hostname.endswith(".hh.kz")
        )
        or re.fullmatch(r"/employer/\d+/?", parsed.path) is None
    ):
        return ""

    return f"https://{hostname}{parsed.path.rstrip('/')}"


async def classify_page(page: Any) -> PageState:
    try:
        wait_for_load_state = getattr(page, "wait_for_load_state", None)
        if wait_for_load_state is not None:
            await wait_for_load_state(
                "domcontentloaded",
                timeout=10_000,
            )
    except Exception as exc:
        logger.warning(
            "page_load_state_failed error=%s",
            type(exc).__name__,
        )

    checks = (
        (
            PageState.CAPTCHA_DETECTED,
            (
                'form[action*="captcha"]',
                '[data-qa="captcha"]',
            ),
        ),
        (
            PageState.ACCESS_DENIED,
            (
                '[data-qa="access-denied"]',
            ),
        ),
        (
            PageState.VACANCY_REMOVED,
            (
                '[data-qa="vacancy-removed"]',
            ),
        ),
        (
            PageState.VACANCY_LOADED,
            (
                '[data-qa="vacancy-description"]',
            ),
        ),
    )

    for state, selectors in checks:
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                is_visible = getattr(locator, "is_visible", None)

                if is_visible is None:
                    continue

                try:
                    visible = await is_visible(timeout=3_000)
                except TypeError:
                    visible = await is_visible()

                if visible:
                    return state

            except RuntimeError as exc:
                logger.debug(
                    "page_selector_check_failed selector=%s error=%s",
                    selector,
                    type(exc).__name__,
                )
                return PageState.NETWORK_ERROR

            except Exception as exc:
                logger.debug(
                    "page_selector_check_failed selector=%s error=%s",
                    selector,
                    type(exc).__name__,
                )

    try:
        title = getattr(page, "title", None)
        if title is not None:
            try:
                page_title = await title()
            except TypeError:
                page_title = ""
        else:
            page_title = ""

        logger.warning("page_structure_changed")
    except Exception as exc:
        logger.warning(
            "page_metadata_read_failed error=%s",
            type(exc).__name__,
        )

    return PageState.PAGE_STRUCTURE_CHANGED


class HHClient:
    def __init__(
        self,
        context: Any,
        settings: Settings,
        database: Database,
        approval_guard: ApprovalGuard,
        telegram_user_id: int,
        *,
        user_settings: UserSettings | None = None,
        hh_account: HHAccount | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        now_factory: Callable[[], datetime] | None = None,
        manual_login_prompt: Callable[[str], Awaitable[None]] | None = None,
        session_lock: asyncio.Lock | None = None,
        session_checker: Callable[[], Awaitable[LoginState]] | None = None,
    ):
        self.context = context
        self.settings = settings
        self.database = database
        self.approval_guard = approval_guard
        self.telegram_user_id = telegram_user_id
        # Phase 5.4: bind the client to the verified HH account.  A
        # supplied account is checked against the database by BOTH its
        # id and telegram_user_id; when omitted, only the caller's own
        # database row is considered.  Legacy callers with no HH account
        # row keep hh_account=None rather than silently borrowing one.
        if hh_account is None:
            hh_account = database.get_hh_account_model(telegram_user_id)
        self.hh_account = (
            verify_hh_account_ownership(
                database,
                telegram_user_id,
                hh_account,
            )
            if hh_account is not None
            else None
        )
        # Stage 4\6: the per-user configuration shared with the owning
        # UserRuntimeContext (identity preserved).  When omitted (legacy
        # callers), materialise a FRESH snapshot from the immutable global
        # defaults — never a shared mutable object.
        self.user_settings = (
            user_settings if user_settings is not None else load_user_settings(settings)
        )
        self.sleep = sleep
        self.now_factory = now_factory or (lambda: datetime.now(UTC))
        self.manual_login_prompt = (
            manual_login_prompt or _console_manual_login_prompt
        )
        # Phase 5.6: one lock per HH session.  Production callers pass the
        # lock owned by the UserRuntimeContext, so check_session(), logout()
        # and submit_application() are serialised for the same account while
        # different accounts never block each other.
        self.session_lock = session_lock or asyncio.Lock()
        # Test seam only: a deterministic session checker.  Production code
        # always inspects the real browser page.
        self.session_checker = session_checker

    async def _delay(self) -> None:
        await self.sleep(
            self.settings.min_seconds_between_actions + random.uniform(0.5, 2.5)
        )

    async def _authorization_state(self, page: Any) -> LoginState:
        """Classify the current page as authorized, unauthorized or unknown.

        Phase 5.5: authorisation requires a POSITIVE signal — an
        authenticated applicant URL together with at least one visible
        logged-in marker.  The absence of login controls is not enough,
        and a page that cannot be inspected is reported as UNKNOWN.
        """
        try:
            page_url = _page_url(page)
            authorized_url = _is_applicant_url(page_url)
            login_url = _is_login_url(page_url)

            has_authorized_marker = await _any_visible(
                page, LOGGED_IN_MARKERS
            )
            has_login_indicator = await _any_visible(
                page, LOGIN_INDICATORS
            )

            # Phase 5.6: contradictory evidence is inconclusive.  A login
            # form together with an authenticated applicant marker must
            # never be reported as a definite logout.
            if (authorized_url or has_authorized_marker) and (
                login_url or has_login_indicator
            ):
                return LoginState.UNKNOWN

            if authorized_url and has_authorized_marker:
                return LoginState.AUTHORIZED

            if login_url or has_login_indicator:
                return LoginState.NOT_AUTHORIZED

            return LoginState.UNKNOWN
        except Exception as exc:
            logger.warning(
                "hh_login_state_check_failed error=%s",
                type(exc).__name__,
            )
            # A technical error is NOT a logout.
            return LoginState.CHECK_FAILED

    async def _manual_login(self, page: Any) -> bool:
        """Wait for a human login, then re-run the strict page check."""
        print("\n" + "=" * 60)
        print("🔑 ТРЕБУЕТСЯ АВТОРИЗАЦИЯ НА HH.KZ")
        print("1. В открывшемся окне браузера войдите в свой аккаунт HH.kz.")
        print("2. После успешного входа вернитесь сюда и нажмите ENTER.")
        print("=" * 60 + "\n")

        try:
            await self.manual_login_prompt(
                "👉 Войдите на HH.kz в браузере и затем нажмите ENTER здесь: "
            )
        except (EOFError, OSError):
            logger.error("hh_manual_login_prompt_unavailable")
            return False

        try:
            await page.goto(
                f"{HH_BASE_URL}/applicant/resumes",
                wait_until="domcontentloaded",
                timeout=90_000,
            )
            await self.sleep(2)
        except Exception as exc:
            logger.warning(
                "hh_manual_login_navigation_failed error=%s",
                type(exc).__name__,
            )
            return False

        state = await self._authorization_state(page)
        if state is not LoginState.AUTHORIZED:
            logger.error(
                "hh_manual_login_not_confirmed state=%s",
                state.value,
            )
            return False

        # Record a successful login only AFTER the page itself has been
        # positively confirmed as authenticated.
        self._mark_authorized(manual=True)
        print("✅ Успешный вход на HH.kz! Запускаем работу агента...\n")
        return True

    async def ensure_login(self) -> bool:
        """Serialised entry point for the startup/manual login flow."""
        async with self.session_lock:
            return await self._ensure_login_unlocked()

    async def _ensure_login_unlocked(self) -> bool:
        """Confirm an existing HH.kz session, allowing manual login if needed.

        Phase 5.5: the persistent browser profile is reused as-is.  This
        method never deletes cookies, local storage or other profile data.
        It returns ``True`` only after a positive authorisation check, so
        the caller can never start the agent loop on an unconfirmed
        session.  Temporary navigation errors keep the existing retry
        behaviour (three attempts with a delay).
        """
        page = await self.context.new_page()
        last_error = SAFE_LOGIN_ERROR_UNVERIFIED
        try:
            for attempt in range(1, 4):
                try:
                    await page.goto(
                        f"{HH_BASE_URL}/applicant/resumes",
                        wait_until="domcontentloaded",
                        timeout=90_000,
                    )
                    await self.sleep(2)
                    state = await self._authorization_state(page)
                except Exception as exc:
                    last_error = SAFE_LOGIN_ERROR_NAVIGATION
                    logger.warning(
                        "hh_login_check_failed attempt=%s error=%s",
                        attempt,
                        type(exc).__name__,
                    )
                    if attempt < 3:
                        await self.sleep(5)
                    continue

                if state is LoginState.AUTHORIZED:
                    self._mark_authorized(manual=False)
                    return True

                if state is LoginState.NOT_AUTHORIZED:
                    if self.settings.browser_headless:
                        logger.error("hh_login_required headless=true")
                        self._mark_login_required(
                            SAFE_LOGIN_ERROR_HEADLESS
                        )
                        return False

                    if await self._manual_login(page):
                        return True

                    self._mark_login_required(SAFE_LOGIN_ERROR_MANUAL)
                    return False

                if state is LoginState.CHECK_FAILED:
                    last_error = SAFE_LOGIN_ERROR_CHECK_FAILED
                    logger.warning(
                        "hh_login_state_check_failed attempt=%s", attempt
                    )
                else:
                    last_error = SAFE_LOGIN_ERROR_UNVERIFIED
                    logger.warning(
                        "hh_login_state_unknown attempt=%s", attempt
                    )
                if attempt < 3:
                    await self.sleep(5)

            # Only a definite login page is "login required".  Repeated
            # UNKNOWN/navigation failures are inconclusive and must keep the
            # prior-authorization status instead of claiming expiry.
            if last_error == SAFE_LOGIN_ERROR_UNVERIFIED:
                self._mark_unknown(last_error)
            else:
                self._mark_check_failed(last_error)
            return False
        finally:
            await page.close()

    def _current_account(self) -> HHAccount | None:
        """Re-read THIS client's account from the database."""
        account = self.hh_account
        if account is None:
            return None
        current = self.database.get_hh_account_by_id(
            account.id, self.telegram_user_id
        )
        if current is None or current.telegram_user_id != self.telegram_user_id:
            return None
        return current

    def _current_status(self) -> str:
        account = self._current_account()
        return account.status if account is not None else ""

    @staticmethod
    def _has_prior_authorization(status: str) -> bool:
        """Persisted status is the memory of a previous positive check."""
        return status in _PRIOR_AUTHORIZATION_STATUSES

    def _write_account(self, operation, *, action: str) -> bool:
        """Apply an ownership-checked write for THIS client's account."""
        if self._current_account() is None:
            logger.warning(
                "hh_account_%s_blocked reason=no_account", action
            )
            return False
        try:
            updated = operation(self.telegram_user_id)
        except Exception as exc:
            logger.warning(
                "hh_account_%s_failed error=%s",
                action,
                type(exc).__name__,
            )
            return False
        if not updated:
            logger.warning("hh_account_%s_failed reason=no_row", action)
            return False
        return True

    def _persist_account_state(self, fields: dict[str, str]) -> bool:
        """Persist ordinary account metadata for THIS verified account only."""
        if "status" in fields or "last_login_at" in fields:
            raise ValueError(
                "status and last_login_at require the centralised HH "
                "authorisation helpers"
            )
        return self._write_account(
            lambda user_id: self.database.update_hh_account(user_id, **fields),
            action="field_update",
        )

    def _transition_account(
        self,
        status: str,
        *,
        last_checked_at: str,
        last_error: str,
    ) -> bool:
        """Apply a centralised non-AUTHORIZED status transition."""
        return self._write_account(
            lambda user_id: self.database.transition_hh_account(
                user_id,
                status,
                last_checked_at=last_checked_at,
                last_error=last_error,
            ),
            action="transition",
        )

    def _confirm_authorized(
        self,
        *,
        last_checked_at: str,
        last_error: str = "",
        login_at: str | None = None,
    ) -> bool:
        """The only client path to AUTHORIZED after a positive check."""
        return self._write_account(
            lambda user_id: self.database.confirm_hh_account_authorized(
                user_id,
                last_checked_at=last_checked_at,
                last_error=last_error,
                last_login_at=login_at,
            ),
            action="authorize",
        )

    def _mark_authorized(self, *, manual: bool) -> None:
        timestamp = _utc_iso(self.now_factory())
        # last_login_at records a real login only.  Automatic re-checks
        # update last_checked_at and the persisted AUTHORIZED status instead.
        self._confirm_authorized(
            last_checked_at=timestamp,
            login_at=timestamp if manual else None,
        )

    def _mark_login_required(self, reason: str) -> None:
        timestamp = _utc_iso(self.now_factory())
        status = self._current_status()
        if status == HH_ACCOUNT_STATUS_LOGGED_OUT:
            target = HH_ACCOUNT_STATUS_LOGGED_OUT
            safe_reason = SAFE_LOGIN_ERROR_LOGGED_OUT
        elif self._has_prior_authorization(status):
            target = HH_ACCOUNT_STATUS_SESSION_EXPIRED
            safe_reason = SAFE_LOGIN_ERROR_SESSION_EXPIRED
        else:
            target = HH_ACCOUNT_STATUS_LOGIN_REQUIRED
            safe_reason = reason
        self._transition_account(
            target,
            last_checked_at=timestamp,
            last_error=safe_reason,
        )

    def _persist_inconclusive(self, state: LoginState, reason: str) -> None:
        """Keep prior-authorization memory while recording a safe error."""
        timestamp = _utc_iso(self.now_factory())
        current_status = self._current_status()
        if self._has_prior_authorization(current_status):
            if current_status == HH_ACCOUNT_STATUS_LOGGED_OUT:
                # Keep the human-readable reason of an explicit logout; an
                # inconclusive follow-up only refreshes the check timestamp.
                reason = SAFE_LOGIN_ERROR_LOGGED_OUT
            self._persist_account_state(
                {
                    "last_checked_at": timestamp,
                    "last_error": reason,
                }
            )
            return
        target = (
            HH_ACCOUNT_STATUS_CHECK_FAILED
            if state is LoginState.CHECK_FAILED
            else HH_ACCOUNT_STATUS_UNKNOWN
        )
        self._transition_account(
            target,
            last_checked_at=timestamp,
            last_error=reason,
        )

    def _mark_check_failed(self, reason: str) -> None:
        self._persist_inconclusive(LoginState.CHECK_FAILED, reason)

    def _mark_unknown(self, reason: str) -> None:
        self._persist_inconclusive(LoginState.UNKNOWN, reason)

    def _persist_check_result(
        self,
        state: LoginState,
        *,
        logout_probe: bool = False,
    ) -> None:
        """Map a check result to a validated, secret-free DB transition."""
        timestamp = _utc_iso(self.now_factory())
        current_status = self._current_status()
        if state is LoginState.AUTHORIZED:
            self._confirm_authorized(last_checked_at=timestamp)
            return

        if state is LoginState.NOT_AUTHORIZED:
            if logout_probe or current_status == HH_ACCOUNT_STATUS_LOGGED_OUT:
                target = HH_ACCOUNT_STATUS_LOGGED_OUT
                reason = SAFE_LOGIN_ERROR_LOGGED_OUT
            elif self._has_prior_authorization(current_status):
                target = HH_ACCOUNT_STATUS_SESSION_EXPIRED
                reason = SAFE_LOGIN_ERROR_SESSION_EXPIRED
            else:
                target = HH_ACCOUNT_STATUS_LOGIN_REQUIRED
                reason = SAFE_LOGIN_ERROR_HEADLESS
            self._transition_account(
                target,
                last_checked_at=timestamp,
                last_error=reason,
            )
            return

        reason = (
            SAFE_LOGIN_ERROR_CHECK_FAILED
            if state is LoginState.CHECK_FAILED
            else SAFE_LOGIN_ERROR_UNVERIFIED
        )
        self._persist_inconclusive(state, reason)

    async def _check_session_unlocked(
        self, *, logout_probe: bool = False
    ) -> LoginState:
        """Inspect the live HH page.  Caller MUST hold ``session_lock``."""
        # Phase 5.6: ownership is verified BEFORE the session checker.
        # A client without a verified HH account must never produce an
        # AUTHORIZED result, so the test seam cannot bypass the account
        # check.  This is the same fail-closed rule the live browser path
        # uses below.
        if self._current_account() is None:
            logger.warning(
                "hh_session_check_without_account user=%s",
                self.telegram_user_id,
            )
            return LoginState.CHECK_FAILED

        if self.session_checker is not None:
            state = await self.session_checker()
            self._persist_check_result(state, logout_probe=logout_probe)
            return state

        page = None
        try:
            page = await self.context.new_page()
            await page.goto(
                f"{HH_BASE_URL}/applicant/resumes",
                wait_until="domcontentloaded",
                timeout=30_000,
            )
            await self.sleep(1)
            state = await self._authorization_state(page)
        except Exception as exc:
            logger.warning(
                "hh_session_check_error error=%s", type(exc).__name__
            )
            state = LoginState.CHECK_FAILED
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception as exc:
                    logger.warning(
                        "hh_session_page_close_failed error=%s",
                        type(exc).__name__,
                    )

        self._persist_check_result(state, logout_probe=logout_probe)
        return state

    async def check_session(self) -> LoginState:
        """Public, lock-protected HH session check."""
        async with self.session_lock:
            return await self._check_session_unlocked()

    async def _invalidate_session_unlocked(self) -> bool:
        """Clear THIS account's local HH session data.

        Caller MUST hold ``session_lock``.  Playwright's BrowserContext has
        ``clear_cookies()`` but no ``clear_storage_state()``; DOM storage is
        cleared by evaluating on a page that has first navigated to the
        HH.kz origin.  ``localStorage`` is never touched on ``about:blank``.
        """
        context = self.context
        if context is None:
            return False

        cookies_cleared = False
        clear_cookies = getattr(context, "clear_cookies", None)
        if callable(clear_cookies):
            try:
                result = clear_cookies()
                if inspect.isawaitable(result):
                    await result
                cookies_cleared = True
            except Exception as exc:
                logger.warning(
                    "hh_logout_cookies_clear_failed error=%s",
                    type(exc).__name__,
                )

        storage_cleared = False
        page = None
        try:
            page = await context.new_page()
            await page.goto(
                f"{HH_BASE_URL}/",
                wait_until="domcontentloaded",
                timeout=30_000,
            )
            if not _is_hh_origin(_page_url(page)):
                raise RuntimeError("storage cleanup requires HH.kz origin")
            evaluate = getattr(page, "evaluate", None)
            if not callable(evaluate):
                raise RuntimeError("page.evaluate is unavailable")
            result = evaluate(
                "localStorage.clear(); sessionStorage.clear();"
            )
            if inspect.isawaitable(result):
                await result
            storage_cleared = True
        except Exception as exc:
            logger.warning(
                "hh_logout_storage_clear_failed error=%s",
                type(exc).__name__,
            )
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception as exc:
                    logger.warning(
                        "hh_logout_page_close_failed error=%s",
                        type(exc).__name__,
                    )
        return cookies_cleared and storage_cleared

    async def logout(self) -> LogoutResult:
        """Log out THIS verified HH account and persist the outcome."""
        async with self.session_lock:
            account = self._current_account()
            if account is None:
                return LogoutResult(
                    False, LoginState.CHECK_FAILED, "no_account"
                )
            try:
                verify_hh_account_ownership(
                    self.database, self.telegram_user_id, account
                )
            except HHAccountOwnershipError:
                logger.error("hh_logout_ownership_rejected")
                return LogoutResult(
                    False, LoginState.CHECK_FAILED, "ownership"
                )

            if not await self._invalidate_session_unlocked():
                # Go through the central persistence path so an already
                # explicit logout is not downgraded to CHECK_FAILED.
                self._persist_check_result(LoginState.CHECK_FAILED)
                return LogoutResult(
                    False, LoginState.CHECK_FAILED, "invalidate_failed"
                )

            state = await self._check_session_unlocked(logout_probe=True)
            if state is LoginState.NOT_AUTHORIZED:
                return LogoutResult(True, state, "confirmed")
            return LogoutResult(False, state, "unconfirmed")

    async def _response_confirmed(self, page: Any, url: str) -> bool:
        try:
            await page.goto(url.split("?")[0], wait_until="domcontentloaded", timeout=30_000)
            await page.locator('[data-qa="vacancy-description"]').wait_for(
                state="visible", timeout=20_000
            )
            response_control = page.locator(
                'a[data-qa="vacancy-response-link-top"], '
                'button[data-qa="vacancy-response-link-top"]'
            ).first
            return await response_control.count() == 0
        except Exception as exc:
            logger.warning(
                "application_confirmation_failed error=%s",
                type(exc).__name__,
            )
            return False

    async def search_vacancies(
        self,
        query: str,
        areas: tuple[str, ...],
        experience_filters: tuple[str, ...],
    ) -> VacancySearchResult:
        results: list[VacancySummary] = []
        found_results = 0
        duplicates = 0
        seen_ids: set[str] = set()
        page = None
        try:
            page = await self.context.new_page()
            for page_number in range(self.settings.max_pages_per_query):
                params: dict[str, Any] = {
                    "text": query,
                    "order_by": "publication_time",
                    "page": page_number,
                }
                if areas:
                    params["area"] = list(areas)
                if experience_filters:
                    params["experience"] = list(experience_filters)
                await self._delay()
                await page.goto(
                    f"{HH_BASE_URL}/search/vacancy?{urlencode(params, doseq=True)}",
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )
                cards = await page.locator('a[data-qa="serp-item__title"]').all()
                if not cards:
                    break
                for card in cards:
                    href = await card.get_attribute("href")
                    title = (await card.inner_text()).strip()

                    if not href:
                        continue

                    if _is_logistics_query(query) and not is_relevant_logistics_title(title):
                        continue

                    job_id = _vacancy_id(href)
                    if not job_id:
                        continue
                    found_results += 1
                    if job_id in seen_ids:
                        duplicates += 1
                        continue
                    seen_ids.add(job_id)
                    existing = self.database.get(job_id)
                    if existing is not None:
                        duplicates += 1
                        if (
                            existing.status is VacancyStatus.PENDING_APPROVAL
                            and len(results) < self.settings.max_vacancies_per_query
                        ):
                            results.append(
                                VacancySummary(job_id, title, href, query, True)
                            )
                        continue
                    if len(results) < self.settings.max_vacancies_per_query:
                        results.append(VacancySummary(job_id, title, href, query))
                if len(results) >= self.settings.max_vacancies_per_query:
                    break
                await self._delay()
            return VacancySearchResult(results, found_results, duplicates)
        except Exception as exc:
            logger.error("vacancy_search_failed query=%r error=%s", query, exc)
            return VacancySearchResult(
                results,
                found_results,
                duplicates,
                f"search_error:{type(exc).__name__}",
            )
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception as exc:
                    logger.warning(
                        "vacancy_search_page_close_failed query=%r error=%s",
                        query,
                        type(exc).__name__,
                    )

    async def read_vacancy(
        self,
        summary: VacancySummary,
        captcha_solver: CaptchaSolver | None = None,
    ) -> VacancyDetails:
        page = None
        try:
            page = await self.context.new_page()
            await self._delay()
            await page.goto(summary.url, wait_until="domcontentloaded", timeout=30_000)
            state = await classify_page(page)
            if state is PageState.CAPTCHA_DETECTED and captcha_solver is not None:
                state = await self._solve_captcha(page, summary, captcha_solver)
            if state is not PageState.VACANCY_LOADED:
                if state is PageState.CAPTCHA_DETECTED:
                    logger.warning("captcha_detected job_id=%s", summary.id)
                return VacancyDetails(summary, state)
            description = (
                await page.locator('[data-qa="vacancy-description"]').inner_text()
            ).strip()
            if not description:
                return VacancyDetails(
                    summary,
                    PageState.PAGE_STRUCTURE_CHANGED,
                    error="vacancy description is empty",
                )
            salary = await _visible_text(
                page,
                '[data-qa="vacancy-salary"]',
            )
            schedule = await _visible_text(
                page,
                '[data-qa="vacancy-schedule"]',
            )
            location = await _visible_text(
                page,
                '[data-qa="vacancy-location"]',
            )
            experience = await _visible_text(
                page,
                '[data-qa="vacancy-experience"]',
            )
            company_locator = page.locator('[data-qa="vacancy-company-name"]').first

            company = (
                (await company_locator.inner_text()).strip()
                if await company_locator.is_visible()
                else ""
            )

            company_link = page.locator('a[href*="/employer/"]').first

            company_url = _company_url(
                summary.url,
                await company_link.get_attribute("href"),
            )
            return VacancyDetails(
                summary=summary,
                state=state,
                company=company,
                description=description,
                company_url=company_url,
                salary=salary,
                schedule=schedule,
                location=location,
                experience=experience,
            )
        except Exception as exc:
            logger.warning(
                "vacancy_read_exception job_id=%s error=%s",
                summary.id,
                type(exc).__name__,
            )
            return VacancyDetails(
                summary,
                PageState.NETWORK_ERROR,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception as exc:
                    logger.warning(
                        "vacancy_page_close_failed job_id=%s error=%s",
                        summary.id,
                        exc,
                    )

    async def read_company_details(self, company_url: str) -> CompanyDetails:
        company_url = _company_url(f"{HH_BASE_URL}/", company_url)
        if not company_url:
            return CompanyDetails()
        page = None
        try:
            page = await self.context.new_page()
            await self._delay()
            await page.goto(company_url, wait_until="domcontentloaded", timeout=30_000)
            rating_text = await _visible_text(
                page, '[data-qa="employer-review-small-widget-total-rating"]'
            )
            reviews_text = await _visible_text(
                page,
                '[data-qa="employer-review-small-widget-review-count-action"]',
            )
            try:
                rating = float(rating_text.replace(",", "."))
                if not 0 <= rating <= 5:
                    rating = None
            except ValueError:
                rating = None
            match = re.search(r"\d[\d\s\u00a0]*", reviews_text)
            reviews_count = (
                int(re.sub(r"\D", "", match.group()))
                if match
                else None
            )
            return CompanyDetails(rating, reviews_count)
        except Exception as exc:
            logger.warning("company_details_read_failed error_type=%s", type(exc).__name__)
            return CompanyDetails()
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception:
                    logger.warning("company_details_page_close_failed")

    async def _solve_captcha(
        self, page: Any, summary: VacancySummary, solver: CaptchaSolver
    ) -> PageState:
        for _ in range(self.settings.captcha_max_attempts):
            screenshot = Path(tempfile.gettempdir()) / f"hh-captcha-{uuid4().hex}.png"
            try:
                await page.screenshot(path=screenshot)
                solution = await solver(
                    screenshot, summary.title, self.settings.captcha_timeout_seconds
                )
                if not solution:
                    return PageState.CAPTCHA_DETECTED
                field = page.locator('input[type="text"]').first
                if not await field.is_visible():
                    return PageState.PAGE_STRUCTURE_CHANGED
                await field.fill(solution)
                submit = page.locator(
                    'button[type="submit"]:visible, button:has-text("Отправить"):visible'
                ).first
                if await submit.is_visible():
                    await submit.click()
                else:
                    await field.press("Enter")
                await self.sleep(1)
                state = await classify_page(page)
                if state is not PageState.CAPTCHA_DETECTED:
                    return state
            finally:
                screenshot.unlink(missing_ok=True)
        return PageState.CAPTCHA_DETECTED

    async def submit_application(self, permission: ApplicationPermission) -> bool:
        """Serialised submit: check and physical send share ONE lock."""
        async with self.session_lock:
            return await self._submit_application_unlocked(permission)

    async def _submit_application_unlocked(
        self, permission: ApplicationPermission
    ) -> bool:
        # The session check and the actual submit happen inside the same
        # critical section.  Never release the lock between them: logout
        # could otherwise invalidate the session before the click.
        state = await self._check_session_unlocked()
        if state is not LoginState.AUTHORIZED:
            logger.warning(
                "application_blocked job_id=%s reason=auth_%s",
                permission.job_id,
                state.value,
            )
            return False

        claim = self.approval_guard.claim(permission)
        if not claim.allowed or claim.vacancy is None:
            logger.warning(
                "application_blocked job_id=%s reason=%s",
                permission.job_id,
                claim.reason,
            )
            return False

        vacancy = claim.vacancy
        page = None

        try:
            page = await self.context.new_page()
            await self._delay()

            await page.goto(
                vacancy.url,
                wait_until="domcontentloaded",
                timeout=30_000,
            )

            response_button = page.locator(
                'a[data-qa="vacancy-response-link-top"], '
                'button[data-qa="vacancy-response-link-top"]'
            ).first

            if not await response_button.is_visible():
                raise RuntimeError("application response control was not found")

            await response_button.click()
            await self._delay()

            # Phase 5.6: never dump raw HH pages to disk.  They can contain
            # CSRF tokens, session-bound markup and personal data.

            # User-scoped resume selection from database
            profile = self.database.get_active_profile(permission.telegram_user_id)
            resume_name = profile.hh.resume_name if profile else ""
            resume_selector = page.locator(
                '[data-qa*="resume-select"], '
                '[data-qa*="resume-selector"]'
            ).first

            if resume_name and await resume_selector.is_visible():
                await resume_selector.click()

                option = page.get_by_text(
                    resume_name,
                    exact=True,
                ).first

                if not await option.is_visible():
                    raise RuntimeError("configured resume was not found")

                await option.click()

            if await page.locator('textarea[name^="task_"]').count() > 0:
                raise QuestionnaireRequiredError("questionnaire_required")

            submit_button = page.locator(
                'button[data-qa*="vacancy-response-submit"]:visible, '
                'button[type="submit"]:visible, '
                'button:has-text("Откликнуться"):visible, '
                'button:has-text("Отправить"):visible'
            ).first

            if not await submit_button.is_visible():
                raise RuntimeError("final application button was not found")

            if not self.database.mark_submit_attempt(
                permission.job_id,
                permission.permit,
                telegram_user_id=permission.telegram_user_id,
                now=self.now_factory(),
                daily_limit=self.user_settings.max_applications_per_day,
            ):
                error = "application permission failed final pre-submit validation"

                self.database.complete_application(
                    permission.job_id,
                    permission.permit,
                    success=False,
                    now=self.now_factory(),
                    error_text=error,
                )

                logger.warning(
                    "application_blocked job_id=%s reason=pre_submit_recheck",
                    permission.job_id,
                )
                return False

            await submit_button.click()

            success_marker = page.locator(
                '[data-qa="vacancy-response-success-standard-notification"], '
                '[data-qa="vacancy-response-success"], '
                '[data-qa="vacancy-response-link-view-topic"]'
            ).or_(
                page.get_by_text(
                    "Отклик отправлен",
                    exact=False,
                )
            ).first

            try:
                await success_marker.wait_for(
                    state="visible",
                    timeout=5_000,
                )
            except Exception:
                raise RuntimeError(
                    "HH.kz did not provide an explicit application success signal"
                )

            if not await self._response_confirmed(
                page,
                vacancy.url,
            ):
                raise RuntimeError(
                    "HH.kz did not confirm the application"
                )

            if not self.database.complete_application(
                permission.job_id,
                permission.permit,
                success=True,
                now=self.now_factory(),
            ):
                raise RuntimeError(
                    "application status could not be completed"
                )

            logger.info(
                "application_sent job_id=%s",
                permission.job_id,
            )

            return True

        except Exception as exc:
            completed = self.database.complete_application(
                permission.job_id,
                permission.permit,
                success=False,
                now=self.now_factory(),
                error_text=str(exc),
            )

            if not completed:
                logger.warning(
                    "application_failure_status_not_updated "
                    "job_id=%s error=%s",
                    permission.job_id,
                    type(exc).__name__,
                )

            # Only the exception TYPE is logged: Playwright errors may
            # contain full URLs with query parameters or session tokens.
            logger.error(
                "application_failed job_id=%s error=%s",
                permission.job_id,
                type(exc).__name__,
            )

            return False

        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception as exc:
                    logger.warning(
                        "application_page_close_failed "
                        "job_id=%s error=%s",
                        permission.job_id,
                        type(exc).__name__,
                    )

    async def check_messages(
        self, notifier: Callable[[str], Awaitable[None]]
    ) -> None:
        page = await self.context.new_page()
        try:
            await self._delay()
            await page.goto(
                f"{HH_BASE_URL}/applicant/negotiations",
                wait_until="domcontentloaded",
                timeout=30_000,
            )
            cards = await page.locator('div[data-qa="negotiations-item"]').all()
            for card in cards:
                badge = card.locator('span[data-qa="negotiations-item-badge"]')
                if not await badge.is_visible():
                    continue
                link = card.locator('a[data-qa="negotiations-item-vacancy-link"]')
                href = await link.get_attribute("href")
                title = (await link.inner_text()).strip()
                message_id = f"{href}:{title}"
                if href and not self.database.is_message_processed(message_id):
                    self.database.add_processed_message(message_id, href, title)
                    await notifier(f"New unread HH message for: {title}\n{HH_BASE_URL}{href}")
        except Exception as exc:
            logger.error(
                "message_check_failed error=%s", type(exc).__name__
            )
        finally:
            await page.close()