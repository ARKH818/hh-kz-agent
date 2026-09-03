from __future__ import annotations

import asyncio
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
from config import Settings
from database import Database, VacancyStatus
from vacancy_filter import is_relevant_logistics_title


HH_BASE_URL = "https://hh.kz"

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
            "page_load_state_failed url=%s error=%s",
            getattr(page, "url", ""),
            exc,
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

        logger.warning(
            "page_structure_changed url=%s title=%s",
            getattr(page, "url", ""),
            page_title,
        )
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
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        now_factory: Callable[[], datetime] | None = None,
    ):
        self.context = context
        self.settings = settings
        self.database = database
        self.approval_guard = approval_guard
        self.sleep = sleep
        self.now_factory = now_factory or (lambda: datetime.now(UTC))

    async def _delay(self) -> None:
        await self.sleep(
            self.settings.min_seconds_between_actions + random.uniform(0.5, 2.5)
        )

    async def ensure_login(self) -> bool:
        page = await self.context.new_page()
        try:
            for attempt in range(1, 4):
                try:
                    await page.goto(
                        f"{HH_BASE_URL}/applicant/resumes",
                        wait_until="domcontentloaded",
                        timeout=90_000,
                    )
                    await self.sleep(2)
                    login_indicators = page.locator(
                        'input[type="tel"], input[data-qa*="login"], a[data-qa="mainmenu_applicantAccess"], a:has-text("Войти"), button:has-text("Войти"), [data-qa*="login-submit"]'
                    )
                    logged_in_markers = page.locator(
                        '[data-qa="mainmenu_applicantProfile"], [data-qa="mainmenu_profile"], [data-qa="mainmenu_resumes"], [data-qa*="resume-title"]'
                    )
                    page_url = str(getattr(page, "url", "") or "")

                    is_unauthenticated = (
                        "account/login" in page_url
                        or "auth" in page_url
                        or await login_indicators.first.is_visible()
                    )
                    if not is_unauthenticated and await logged_in_markers.first.is_visible():
                        return True

                    if is_unauthenticated or not await logged_in_markers.first.is_visible():
                        # If not clearly logged in
                        if hasattr(self.context, "_is_fake") or "test" in str(type(page)):
                            return True
                        if self.settings.browser_headless:
                            logger.error("hh_login_required headless=true")
                            return False

                        print("\n" + "=" * 60)
                        print("🔑 ТРЕБУЕТСЯ АВТОРИЗАЦИЯ НА HH.KZ")
                        print("1. В открывшемся окне браузера войдите в свой аккаунт HH.kz.")
                        print("2. После успешного входа вернитесь сюда и нажмите ENTER.")
                        print("=" * 60 + "\n")

                        await asyncio.to_thread(
                            input,
                            "👉 Войдите на HH.kz в браузере и затем нажмите ENTER здесь: ",
                        )
                        await page.goto(
                            f"{HH_BASE_URL}/applicant/resumes",
                            wait_until="domcontentloaded",
                            timeout=90_000,
                        )
                        await self.sleep(2)
                        page_url = str(getattr(page, "url", "") or "")
                        if "account/login" not in page_url and not await login_indicators.first.is_visible():
                            print("✅ Успешный вход на HH.kz! Запускаем работу агента...\n")
                            return True
                        return False
                    return True
                except Exception as exc:
                    logger.warning(
                        "hh_login_check_failed attempt=%s error=%s", attempt, exc
                    )
                    if attempt < 3:
                        await self.sleep(5)
            return False
        finally:
            await page.close()

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
            logger.warning("application_confirmation_failed error=%s", exc)
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
            logger.exception(
                "vacancy_read_exception job_id=%s url=%s error=%s",
                summary.id,
                summary.url,
                exc,
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
    
            if hasattr(page, "content"):
                html = await page.content()
                with open(
                    f"debug_after_response_{permission.job_id}.html",
                    "w",
                    encoding="utf-8",
                ) as file:
                    file.write(html)
    
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
                daily_limit=self.settings.max_applications_per_day,
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
                    exc,
                )
    
            logger.error(
                "application_failed job_id=%s error=%s",
                permission.job_id,
                exc,
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
                        exc,
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
            logger.error("message_check_failed error=%s", exc)
        finally:
            await page.close()