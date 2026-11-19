import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from approval import ApplicationPermission, ApprovalGuard
from config import load_settings, load_user_settings
from database import Database, VacancyStatus
from hh_client import (
    HH_ACCOUNT_STATUS_AUTHORIZED,
    HH_ACCOUNT_STATUS_CHECK_FAILED,
    HH_ACCOUNT_STATUS_LOGGED_OUT,
    HH_ACCOUNT_STATUS_LOGIN_REQUIRED,
    HH_ACCOUNT_STATUS_SESSION_EXPIRED,
    HH_ACCOUNT_STATUS_UNKNOWN,
    HHClient,
    LoginState,
    PageState,
    VacancySummary,
    _is_hh_origin,
    classify_page,
)
from runtime_context import HHAccountOwnershipError
from tests.test_config import VALID_ENV, write_profile
from vacancy_filter import title_rejection_reason



NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)


async def authorized_session() -> LoginState:
    """Test seam: a positive session re-check before a physical submit."""
    return LoginState.AUTHORIZED


class FakeLocator:
    def __init__(self, visible: bool, page: "ResumeTrackingPage | None" = None, selector_type: str = ""):
        self.visible = visible
        self.first = self
        self._page = page
        self._selector_type = selector_type

    async def is_visible(self) -> bool:
        return self.visible

    async def inner_text(self) -> str:
        return ""

    def or_(self, _other):
        return self

    async def count(self) -> int:
        return 1 if self.visible else 0

    async def click(self) -> None:
        if self._page is not None:
            self._page._on_click(self._selector_type)

    async def wait_for(self, *, state: str, timeout: int) -> None:
        if not self.visible:
            raise RuntimeError("element not visible")


class FakePage:
    def __init__(self, visible_selector: str = "", broken: bool = False):
        self.visible_selector = visible_selector
        self.broken = broken

    def locator(self, selector: str) -> FakeLocator:
        if self.broken:
            raise RuntimeError("page closed")
        return FakeLocator(selector == self.visible_selector)

    async def goto(self, url: str, **kwargs) -> None:
        return None

    async def close(self) -> None:
        return None


class FakeContext:
    def __init__(self, page: FakePage):
        self.page = page

    async def new_page(self) -> FakePage:
        return self.page


class SearchCard:
    def __init__(self, href: str, title: str):
        self.href = href
        self.title = title

    async def get_attribute(self, name: str) -> str | None:
        return self.href if name == "href" else None

    async def inner_text(self) -> str:
        return self.title


class TextLocator(FakeLocator):
    def __init__(self, text: str = "", href: str | None = None):
        super().__init__(bool(text))
        self.text = text
        self.href = href

    async def inner_text(self) -> str:
        return self.text

    async def get_attribute(self, name: str) -> str | None:
        return self.href if name == "href" else None


class SearchLocator:
    def __init__(self, cards: list[SearchCard]):
        self.cards = cards

    async def all(self) -> list[SearchCard]:
        return self.cards


class SearchPage(FakePage):
    def __init__(self, cards: list[SearchCard] | None = None):
        super().__init__()
        self.cards = cards or []

    def locator(self, selector: str) -> SearchLocator:
        return SearchLocator(self.cards)


class VacancyPage(FakePage):
    def locator(self, selector: str) -> TextLocator:
        values = {
            '[data-qa="vacancy-description"]': "Описание вакансии.",
            '[data-qa="vacancy-company-name"]': "Компания",
        }

        if selector == 'a[href*="/employer/"]':
            return TextLocator(
                "Компания",
                "/employer/123?hhtmFrom=vacancy",
            )

        return TextLocator(values.get(selector, ""))


class EmployerPage(FakePage):
    def __init__(self, rating: str = "", reviews: str = ""):
        super().__init__()
        self.values = {
            '[data-qa="employer-review-small-widget-total-rating"]': rating,
            '[data-qa="employer-review-small-widget-review-count-action"]': reviews,
        }

    def locator(self, selector: str) -> TextLocator:
        return TextLocator(self.values.get(selector, ""))


class FailingContext:
    async def new_page(self):
        raise RuntimeError("page creation failed")


class LoginPage(FakePage):
    """Fake page exposing a URL and a set of visible selectors."""

    def __init__(
        self,
        *,
        url: str = "",
        visible_selectors: tuple[str, ...] = (),
    ):
        super().__init__()
        self.url = url
        self.visible_selectors = set(visible_selectors)
        self.goto_urls: list[str] = []
        self.closed = False

    def locator(self, selector: str) -> FakeLocator:
        return FakeLocator(selector in self.visible_selectors)

    async def goto(self, url: str, **kwargs) -> None:
        self.goto_urls.append(url)

    async def close(self) -> None:
        self.closed = True


class RetryLoginPage(LoginPage):
    def __init__(self):
        super().__init__(
            url="https://hh.kz/applicant/resumes",
            visible_selectors=('[data-qa="mainmenu_applicantProfile"]',),
        )
        self.attempts = 0

    async def goto(self, url: str, **kwargs) -> None:
        self.attempts += 1
        if self.attempts < 3:
            raise RuntimeError("temporary navigation failure")


class CaptchaLocator:
    def __init__(self, page: "CaptchaPage"):
        self.page = page
        self.first = self

    async def is_visible(self) -> bool:
        return True

    async def fill(self, value: str) -> None:
        self.page.solution = value

    async def press(self, key: str) -> None:
        self.page.actions.append(f"press:{key}")

    async def click(self) -> None:
        self.page.actions.append("click:submit")
        self.page.solved = True


class CaptchaPage:
    def __init__(self):
        self.actions: list[str] = []
        self.solution = ""
        self.solved = False

    def locator(self, selector: str):
        if selector == 'input[type="text"]' or "button" in selector:
            return CaptchaLocator(self)
        return FakeLocator(
            (self.solved and selector == '[data-qa="vacancy-description"]')
            or (not self.solved and selector == 'form[action*="captcha"]')
        )

    async def screenshot(self, *, path: Path) -> None:
        path.touch()


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ('[data-qa="vacancy-description"]', PageState.VACANCY_LOADED),
        ('form[action*="captcha"]', PageState.CAPTCHA_DETECTED),
        ('[data-qa="access-denied"]', PageState.ACCESS_DENIED),
        ('[data-qa="vacancy-removed"]', PageState.VACANCY_REMOVED),
        ("", PageState.PAGE_STRUCTURE_CHANGED),
    ],
)
def test_page_state_uses_explicit_signals(selector: str, expected: PageState) -> None:
    assert asyncio.run(classify_page(FakePage(selector))) is expected


def test_page_state_reports_network_error() -> None:
    assert asyncio.run(classify_page(FakePage(broken=True))) is PageState.NETWORK_ERROR


def test_visible_but_empty_description_reports_structure_change(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FakeContext(FakePage('[data-qa="vacancy-description"]')),
        settings,
        database,
        ApprovalGuard(settings, database),
        telegram_user_id=42,
    )
    summary = VacancySummary(
        "job-1", "Developer", "https://example.com/vacancy/job-1", "Python"
    )

    result = asyncio.run(client.read_vacancy(summary))

    assert result.state is PageState.PAGE_STRUCTURE_CHANGED


def test_page_creation_failure_reports_network_error(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FailingContext(), settings, database, ApprovalGuard(settings, database),
        telegram_user_id=42,
    )
    summary = VacancySummary(
        "job-1", "Developer", "https://example.com/vacancy/job-1", "Python"
    )

    result = asyncio.run(client.read_vacancy(summary))

    assert result.state is PageState.NETWORK_ERROR
    assert "page creation failed" in result.error


def test_search_page_creation_failure_returns_safe_error(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FailingContext(), settings, database, ApprovalGuard(settings, database),
        telegram_user_id=42,
    )

    result = asyncio.run(client.search_vacancies("Python", (), ()))

    assert result.summaries == []
    assert result.error_reason == "search_error:RuntimeError"


def test_search_ignores_ad_redirects_without_numeric_vacancy_id(
    tmp_path: Path,
) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
        max_pages_per_query=1,
    )
    database = Database(settings.database_path)
    database.init()
    page = SearchPage(
        [
            SearchCard(
                "https://adsrv.hh.ru/click?clickType=link_to_vacancy",
                "Ad",
            ),
            SearchCard("https://hh.ru/vacancy/135006927?from=search", "DevOps"),
        ]
    )

    async def no_sleep(_seconds: float) -> None:
        return None

    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=no_sleep,
        telegram_user_id=42,
    )

    result = asyncio.run(client.search_vacancies("DevOps", (), ()))

    assert [(item.id, item.title) for item in result.summaries] == [
        ("135006927", "DevOps")
    ]


def test_search_counts_found_new_and_existing_results(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
        max_pages_per_query=1,
    )
    database = Database(settings.database_path)
    database.init()
    assert database.discover(
        job_id="135006927",
        title="Existing",
        company="Example",
        url="https://hh.ru/vacancy/135006927",
        description_hash="hash",
        search_query="DevOps",
        discovered_at=datetime(2026, 7, 29, tzinfo=UTC),
    )
    page = SearchPage(
        [
            SearchCard("https://hh.ru/vacancy/135006927", "Existing"),
            SearchCard("https://hh.ru/vacancy/135006928", "New"),
            SearchCard("https://example.com/ad", "Ad"),
        ]
    )

    async def no_sleep(_seconds: float) -> None:
        return None

    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=no_sleep,
        telegram_user_id=42,
    )

    result = asyncio.run(client.search_vacancies("DevOps", (), ()))

    assert result.found_results == 2
    assert result.duplicates == 1
    assert [item.id for item in result.summaries] == ["135006928"]


def test_search_returns_only_pending_existing_vacancies_as_repeats(
    tmp_path: Path,
) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
        max_pages_per_query=1,
    )
    database = Database(settings.database_path)
    database.init()
    now = NOW
    for job_id in ("135006927", "135006928"):
        assert database.discover(
            job_id=job_id,
            title="Existing",
            company="Example",
            url=f"https://hh.ru/vacancy/{job_id}",
            description_hash="hash",
            search_query="DevOps",
            discovered_at=now,
        )
        assert database.request_approval(
            job_id=job_id,
            now=now,
        )
    assert database.skip("135006928", 42, 42)
    page = SearchPage(
        [
            SearchCard("https://hh.ru/vacancy/135006927", "Pending"),
            SearchCard("https://hh.ru/vacancy/135006928", "Skipped"),
            SearchCard("https://hh.ru/vacancy/135006929", "New"),
        ]
    )

    async def no_sleep(_seconds: float) -> None:
        return None

    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=no_sleep,
        telegram_user_id=42,
    )

    result = asyncio.run(client.search_vacancies("DevOps", (), ()))

    assert [(item.id, item.previously_sent) for item in result.summaries] == [
        ("135006927", True),
        ("135006929", False),
    ]
    assert result.duplicates == 2


def test_read_vacancy_collects_safe_company_url(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FakeContext(VacancyPage()),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=lambda _: asyncio.sleep(0),
        telegram_user_id=42,
    )

    result = asyncio.run(
        client.read_vacancy(
            VacancySummary("job-1", "Developer", "https://hh.kz/vacancy/1", "Python")
        )
    )

    assert result.company_url == "https://hh.kz/employer/123"


def test_read_company_details_parses_rating_and_review_count(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FakeContext(EmployerPage("3,4", "14 отзывов")),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=lambda _: asyncio.sleep(0),
        telegram_user_id=42,
    )

    result = asyncio.run(client.read_company_details("https://hh.kz/employer/123"))

    assert result.rating == 3.4
    assert result.reviews_count == 14


def test_read_company_details_rejects_non_hh_url(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FakeContext(EmployerPage("5", "100 отзывов")),
        settings,
        database,
        ApprovalGuard(settings, database),
        telegram_user_id=42,
    )

    result = asyncio.run(client.read_company_details("https://example.com/employer/1"))

    assert result.rating is None
    assert result.reviews_count is None


def test_login_check_retries_temporary_navigation_errors(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    page = RetryLoginPage()

    async def no_sleep(_seconds: float) -> None:
        return None

    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=no_sleep,
        telegram_user_id=42,
    )

    assert asyncio.run(client.ensure_login())
    assert page.attempts == 3


def test_captcha_solution_uses_submit_button(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    page = CaptchaPage()
    client = HHClient(
        FakeContext(page), settings, database, ApprovalGuard(settings, database),
        telegram_user_id=42,
    )
    summary = VacancySummary(
        "job-1", "Developer", "https://example.com/vacancy/job-1", "Python"
    )

    async def solve(*_args):
        return "abcd"

    result = asyncio.run(client._solve_captcha(page, summary, solve))

    assert result is PageState.VACANCY_LOADED
    assert page.solution == "abcd"
    assert page.actions == ["click:submit"]


@pytest.mark.parametrize(
    ("title", "excluded", "expected"),
    [
        ("Senior Python developer", (), "senior"),
        ("Python sales engineer", ("sales",), "sales"),
        ("Python developer", (), None),
    ],
)
def test_title_filter_returns_the_matched_reason(
    title: str, excluded: tuple[str, ...], expected: str | None
) -> None:
    assert title_rejection_reason(title, excluded) == expected

# ============================================================================
# Stage 3.6: User-scoped resume tests
# ============================================================================

class ResumeTrackingLocator(FakeLocator):
    """Locator that tracks which resume name was looked up."""
    def __init__(self, visible: bool, text: str = "", page: "ResumeTrackingPage | None" = None):
        super().__init__(visible, page=page, selector_type="resume_option")
        self._text = text

    async def inner_text(self) -> str:
        return self._text


class ResumeTrackingPage:
    """Stateful fake page that simulates HH application flow."""

    _STATE_INITIAL = "initial"
    _STATE_RESPONSE_CLICKED = "response_clicked"
    _STATE_RESUME_SELECTED = "resume_selected"
    _STATE_SUBMITTED = "submitted"

    def __init__(self, expected_resume: str = ""):
        self.actions: list[str] = []
        self._expected_resume = expected_resume
        self._state = self._STATE_INITIAL

    def _on_click(self, selector_type: str = "") -> None:
        """Transition state on click.

        selector_type hints which element was clicked:
        - "response": vacancy response link
        - "resume_selector": resume dropdown
        - "resume_option": specific resume option
        - "submit": final submit button
        """
        if self._state == self._STATE_INITIAL and selector_type in ("", "response"):
            self._state = self._STATE_RESPONSE_CLICKED
        elif self._state == self._STATE_RESPONSE_CLICKED and selector_type in ("", "resume_option"):
            self._state = self._STATE_RESUME_SELECTED
        elif self._state == self._STATE_RESUME_SELECTED and selector_type in ("", "submit"):
            self._state = self._STATE_SUBMITTED

    def locator(self, selector: str):
        # Response link: visible only in initial state
        if "vacancy-response-link-top" in selector:
            return FakeLocator(self._state == self._STATE_INITIAL, page=self, selector_type="response")

        # Resume selector: visible after response clicked
        if "resume-select" in selector or "resume-selector" in selector:
            return FakeLocator(self._state == self._STATE_RESPONSE_CLICKED, page=self, selector_type="resume_selector")

        # Questionnaire: never visible in our tests
        if selector == 'textarea[name^="task_"]':
            return FakeLocator(False, page=self)

        # Submit button: visible after resume selected
        if "vacancy-response-submit" in selector or "submit" in selector.lower():
            return FakeLocator(self._state == self._STATE_RESUME_SELECTED, page=self, selector_type="submit")

        # Success marker: visible after submitted
        if "vacancy-response-success" in selector:
            return FakeLocator(self._state == self._STATE_SUBMITTED, page=self)

        # Vacancy description: visible after submitted (for _response_confirmed)
        if "vacancy-description" in selector:
            return FakeLocator(self._state == self._STATE_SUBMITTED, page=self)

        return FakeLocator(False, page=self)

    def get_by_text(self, text: str, exact: bool = False):
        if text == self._expected_resume and self._state == self._STATE_RESPONSE_CLICKED:
            self.actions.append(f"selected_resume:{text}")
            return ResumeTrackingLocator(True, text, page=self)
        return ResumeTrackingLocator(False, text)

    async def goto(self, url: str, **kwargs) -> None:
        return None

    async def close(self) -> None:
        return None



def _approval_user_settings(settings):
    """Explicit approval-enabled UserSettings snapshot for Stage 3.6 tests."""
    return load_user_settings(
        settings,
        app_mode="approval",
        enable_real_apply=True,
        max_applications_per_day=10,
        approval_ttl_minutes=30,
    )

def _setup_user_with_resume(
    database: Database,
    telegram_user_id: int,
    resume_title: str,
    now: datetime,
) -> None:
    """Helper to create a user with an active resume."""
    database.get_or_create_user(
        telegram_user_id=telegram_user_id,
        username=f"user{telegram_user_id}",
        now=now,
    )
    database.create_resume(
        telegram_user_id=telegram_user_id,
        hh_resume_id=f"resume-{telegram_user_id}",
        title=resume_title,
        is_active=True,
        now=now,
    )
    # Phase 5.6: the client only returns AUTHORIZED for an owned HH
    # account, so the resume tests must set one up (idempotent when a
    # test already created it explicitly).
    database.get_or_create_hh_account(telegram_user_id=telegram_user_id, now=now)


def _setup_vacancy_and_approve(
    database: Database,
    job_id: str,
    telegram_user_id: int,
    now: datetime,
) -> str:
    """Helper to discover, request approval, and approve a vacancy."""
    database.discover(
        job_id=job_id,
        title="Test Job",
        company="Test Co",
        url=f"https://example.com/vacancy/{job_id}",
        description_hash="hash",
        search_query="Python",
        discovered_at=now,
    )
    database.request_approval(job_id=job_id, now=now)
    permit = database.approve(
        job_id=job_id,
        telegram_user_id=telegram_user_id,
        expected_user_id=telegram_user_id,
        now=now + timedelta(minutes=1),
        ttl_minutes=30,
    )
    assert permit is not None
    return permit


def test_user_42_uses_only_own_resume(tmp_path: Path) -> None:
    """User 42's application must use user 42's resume from DB."""
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
        app_mode="approval",
        enable_real_apply=True,
        max_applications_per_day=10,
        approval_ttl_minutes=30,
    )
    database = Database(settings.database_path)
    database.init()
    now = NOW

    user_settings = _approval_user_settings(settings)

    _setup_user_with_resume(database, 42, "Resume FortyTwo", now)
    permit = _setup_vacancy_and_approve(database, "job-42", 42, now)

    page = ResumeTrackingPage(expected_resume="Resume FortyTwo")
    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(user_settings, database, now_factory=lambda: NOW),
        sleep=lambda _: asyncio.sleep(0),
        now_factory=lambda: NOW,
        session_checker=authorized_session,
        telegram_user_id=42,
        user_settings=user_settings,
    )

    permission = ApplicationPermission("job-42", permit, 42)
    result = asyncio.run(client.submit_application(permission))

    assert result is True
    assert "selected_resume:Resume FortyTwo" in page.actions


def test_user_43_uses_only_own_resume(tmp_path: Path) -> None:
    """User 43's application must use user 43's resume from DB."""
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
        app_mode="approval",
        enable_real_apply=True,
        max_applications_per_day=10,
        approval_ttl_minutes=30,
    )
    database = Database(settings.database_path)
    database.init()
    now = NOW

    user_settings = _approval_user_settings(settings)

    _setup_user_with_resume(database, 43, "Resume FortyThree", now)
    permit = _setup_vacancy_and_approve(database, "job-43", 43, now)

    page = ResumeTrackingPage(expected_resume="Resume FortyThree")
    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(user_settings, database, now_factory=lambda: NOW),
        sleep=lambda _: asyncio.sleep(0),
        now_factory=lambda: NOW,
        session_checker=authorized_session,
        telegram_user_id=43,
        user_settings=user_settings,
    )

    permission = ApplicationPermission("job-43", permit, 43)
    result = asyncio.run(client.submit_application(permission))

    assert result is True
    assert "selected_resume:Resume FortyThree" in page.actions


def test_user_42_cannot_use_user_43_resume(tmp_path: Path) -> None:
    """User 42's application must NOT select user 43's resume."""
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
        app_mode="approval",
        enable_real_apply=True,
        max_applications_per_day=10,
        approval_ttl_minutes=30,
    )
    database = Database(settings.database_path)
    database.init()
    now = NOW

    user_settings = _approval_user_settings(settings)

    # User 42 has resume "Resume FortyTwo"
    _setup_user_with_resume(database, 42, "Resume FortyTwo", now)
    # User 43 has resume "Resume FortyThree"
    _setup_user_with_resume(database, 43, "Resume FortyThree", now)

    permit = _setup_vacancy_and_approve(database, "job-42", 42, now)

    # Page only shows user 43's resume (simulating wrong resume on HH page)
    page = ResumeTrackingPage(expected_resume="Resume FortyThree")
    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(user_settings, database, now_factory=lambda: NOW),
        sleep=lambda _: asyncio.sleep(0),
        now_factory=lambda: NOW,
        session_checker=authorized_session,
        telegram_user_id=42,
        user_settings=user_settings,
    )

    permission = ApplicationPermission("job-42", permit, 42)
    # Should fail because user 42's resume is not found on the page
    result = asyncio.run(client.submit_application(permission))

    assert result is False
    # Should NOT have selected user 43's resume
    assert "selected_resume:Resume FortyThree" not in page.actions


def test_missing_resume_does_not_fall_back_to_global_profile(tmp_path: Path) -> None:
    """If user has no resume in DB, application should not use global profile resume."""
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
        app_mode="approval",
        enable_real_apply=True,
        max_applications_per_day=10,
        approval_ttl_minutes=30,
    )
    database = Database(settings.database_path)
    database.init()
    now = NOW

    user_settings = _approval_user_settings(settings)

    # User exists but has NO resume
    database.get_or_create_user(
        telegram_user_id=42,
        username="user42",
        now=now,
    )

    permit = _setup_vacancy_and_approve(database, "job-42", 42, now)

    # Page has no resume selector visible
    page = ResumeTrackingPage(expected_resume="")
    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(user_settings, database, now_factory=lambda: NOW),
        sleep=lambda _: asyncio.sleep(0),
        now_factory=lambda: NOW,
        session_checker=authorized_session,
        telegram_user_id=42,
        user_settings=user_settings,
    )

    permission = ApplicationPermission("job-42", permit, 42)
    # Should proceed without resume selection (resume_name is empty)
    result = asyncio.run(client.submit_application(permission))

    # The key assertion: no resume was selected because user has none
    assert "selected_resume:" not in str(page.actions)


# ============================================================================
# Phase 5.4 — HHClient ownership binding
# ============================================================================


def _ownership_settings(tmp_path: Path):
    return replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )


def _setup_hh_accounts(database: Database):
    database.create_user(telegram_user_id=42, now=NOW)
    database.create_user(telegram_user_id=43, now=NOW)
    account_42_id = database.create_hh_account(
        telegram_user_id=42,
        account_label="Account 42",
        browser_profile_dir="profiles/user_42",
        now=NOW,
    )
    account_43_id = database.create_hh_account(
        telegram_user_id=43,
        account_label="Account 43",
        browser_profile_dir="profiles/user_43",
        now=NOW,
    )
    account_42 = database.get_hh_account_by_id(account_42_id, 42)
    account_43 = database.get_hh_account_by_id(account_43_id, 43)
    assert account_42 is not None
    assert account_43 is not None
    return account_42, account_43


def test_hh_client_accepts_verified_own_hh_account(tmp_path: Path) -> None:
    """A database-backed account for the same Telegram user is bound."""
    settings = _ownership_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    user_settings = _approval_user_settings(settings)
    guard = ApprovalGuard(user_settings, database, now_factory=lambda: NOW)

    client = HHClient(
        FakeContext(FakePage()),
        settings,
        database,
        guard,
        telegram_user_id=42,
        user_settings=user_settings,
        hh_account=account_42,
    )

    assert client.hh_account is not None
    assert client.hh_account.id == account_42.id
    assert client.hh_account.telegram_user_id == 42
    assert client.approval_guard is guard
    assert client.user_settings is user_settings


def test_hh_client_loads_and_verifies_own_hh_account(tmp_path: Path) -> None:
    """When no account is supplied, only the caller's own row is loaded."""
    settings = _ownership_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)

    client = HHClient(
        FakeContext(FakePage()),
        settings,
        database,
        ApprovalGuard(settings, database),
        telegram_user_id=42,
    )

    assert client.hh_account is not None
    assert client.hh_account.id == account_42.id
    assert client.hh_account.telegram_user_id == 42


def test_hh_client_rejects_another_users_hh_account(tmp_path: Path) -> None:
    """User 42 cannot construct an HHClient bound to user 43's account."""
    settings = _ownership_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    _, account_43 = _setup_hh_accounts(database)

    with pytest.raises(HHAccountOwnershipError, match="does not belong"):
        HHClient(
            FakeContext(FakePage()),
            settings,
            database,
            ApprovalGuard(settings, database),
            telegram_user_id=42,
            hh_account=account_43,
        )


def test_hh_client_rejects_forged_hh_account_id(tmp_path: Path) -> None:
    """A forged id is rejected by the database ownership lookup."""
    settings = _ownership_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, account_43 = _setup_hh_accounts(database)
    forged = replace(account_42, id=account_43.id)

    with pytest.raises(HHAccountOwnershipError, match="does not exist"):
        HHClient(
            FakeContext(FakePage()),
            settings,
            database,
            ApprovalGuard(settings, database),
            telegram_user_id=42,
            hh_account=forged,
        )


def test_hh_client_rejects_forged_browser_profile(tmp_path: Path) -> None:
    """The client refuses a profile path not stored on its account row."""
    settings = _ownership_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    forged = replace(
        account_42,
        browser_profile_dir="profiles/foreign-user",
    )

    with pytest.raises(HHAccountOwnershipError, match="browser profile"):
        HHClient(
            FakeContext(FakePage()),
            settings,
            database,
            ApprovalGuard(settings, database),
            telegram_user_id=42,
            hh_account=forged,
        )


# ============================================================================
# Phase 5.5 - persistent session reuse and strict login verification
# ============================================================================

AUTHORIZED_PAGE_URL = "https://hh.kz/applicant/resumes"
LOGIN_PAGE_URL = (
    "https://hh.kz/account/login?backurl=%2Fapplicant%2Fresumes"
)
AUTHORIZED_MARKER = '[data-qa="mainmenu_applicantProfile"]'
LOGIN_MARKER = 'input[type="tel"]'


def _login_settings(tmp_path: Path, *, headless: bool = False):
    return replace(
        _ownership_settings(tmp_path),
        browser_headless=headless,
    )


def _login_client(
    database: Database,
    settings,
    account,
    page: LoginPage,
    *,
    headless: bool = False,
    now: datetime = NOW,
    prompt=None,
):
    effective = replace(settings, browser_headless=headless)
    user_settings = _approval_user_settings(effective)
    return HHClient(
        FakeContext(page),
        effective,
        database,
        ApprovalGuard(user_settings, database, now_factory=lambda: now),
        telegram_user_id=42,
        user_settings=user_settings,
        hh_account=account,
        sleep=lambda _seconds: asyncio.sleep(0),
        now_factory=lambda: now,
        manual_login_prompt=prompt,
    )


def test_ensure_login_confirms_positive_authorization_signals(
    tmp_path: Path,
) -> None:
    """A real authenticated page updates status and clears stale errors."""
    settings = _login_settings(tmp_path, headless=True)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    assert database.update_hh_account(42, last_error="stale error")
    page = LoginPage(
        url=AUTHORIZED_PAGE_URL,
        visible_selectors=(AUTHORIZED_MARKER,),
    )
    client = _login_client(database, settings, account_42, page)

    assert asyncio.run(client.ensure_login()) is True

    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_AUTHORIZED
    assert stored.last_checked_at == NOW.isoformat()
    assert stored.last_error == ""
    # Automatic positive checks update last_checked_at and the AUTHORIZED
    # status, not last_login_at (which is reserved for a real login).
    assert stored.last_login_at is None
    assert page.closed is True


def test_ensure_login_rejects_login_page_in_headless_mode(
    tmp_path: Path,
) -> None:
    """An explicit login page is never accepted as an authorized session."""
    settings = _login_settings(tmp_path, headless=True)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    page = LoginPage(
        url=LOGIN_PAGE_URL,
        visible_selectors=(LOGIN_MARKER,),
    )
    client = _login_client(database, settings, account_42, page)

    assert asyncio.run(client.ensure_login()) is False

    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_LOGIN_REQUIRED
    assert stored.last_checked_at == NOW.isoformat()
    assert stored.last_error
    assert stored.last_login_at is None
    assert "password" not in stored.last_error.casefold()
    assert "cookie" not in stored.last_error.casefold()


def test_ensure_login_rejects_absence_of_login_indicators_as_unknown(
    tmp_path: Path,
) -> None:
    """No login controls but also no positive marker is NOT a success."""
    settings = _login_settings(tmp_path, headless=True)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    page = LoginPage(url=AUTHORIZED_PAGE_URL)
    client = _login_client(database, settings, account_42, page)

    assert asyncio.run(client.ensure_login()) is False

    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_UNKNOWN
    assert "could not be verified" in stored.last_error.casefold()
    assert stored.last_login_at is None
    assert len(page.goto_urls) == 3


def test_ensure_login_has_no_fake_page_bypass(tmp_path: Path) -> None:
    """The old in-test bypass (fake context / test class name) is gone."""
    settings = _login_settings(tmp_path, headless=True)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    page = LoginPage(url=AUTHORIZED_PAGE_URL)
    context = FakeContext(page)
    context._is_fake = True  # type: ignore[attr-defined]
    client = _login_client(database, settings, account_42, page)
    client.context = context

    assert asyncio.run(client.ensure_login()) is False
    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_UNKNOWN


def test_manual_login_requires_positive_recheck(tmp_path: Path) -> None:
    """Manual login is accepted only after a positive page re-check."""
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    page = LoginPage(
        url=LOGIN_PAGE_URL,
        visible_selectors=(LOGIN_MARKER,),
    )
    prompts: list[str] = []

    async def prompt(message: str) -> None:
        prompts.append(message)
        page.url = AUTHORIZED_PAGE_URL
        page.visible_selectors = {AUTHORIZED_MARKER}

    client = _login_client(
        database,
        settings,
        account_42,
        page,
        headless=False,
        prompt=prompt,
    )

    assert asyncio.run(client.ensure_login()) is True

    assert len(prompts) == 1
    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_AUTHORIZED
    assert stored.last_login_at == NOW.isoformat()
    assert stored.last_checked_at == NOW.isoformat()
    assert stored.last_error == ""
    assert page.url == AUTHORIZED_PAGE_URL


def test_ensure_login_inconclusive_result_does_not_claim_expiry(
    tmp_path: Path,
) -> None:
    """Repeated UNKNOWN checks are not proof that a session expired."""
    settings = _login_settings(tmp_path, headless=True)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database, last_login_at=None)
    client = _login_client(
        database, settings, account_42, FakePage(), headless=True
    )

    assert asyncio.run(client.ensure_login()) is False

    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_AUTHORIZED
    assert "could not be verified" in stored.last_error.casefold()
    assert "expired" not in stored.last_error.casefold()
    assert stored.last_login_at is None


def test_manual_login_without_positive_marker_is_rejected(
    tmp_path: Path,
) -> None:
    """Pressing ENTER alone must not create a successful login record."""
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    page = LoginPage(
        url=LOGIN_PAGE_URL,
        visible_selectors=(LOGIN_MARKER,),
    )
    prompts: list[str] = []

    async def prompt(message: str) -> None:
        prompts.append(message)

    client = _login_client(
        database,
        settings,
        account_42,
        page,
        headless=False,
        prompt=prompt,
    )

    assert asyncio.run(client.ensure_login()) is False

    assert len(prompts) == 1
    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_LOGIN_REQUIRED
    assert stored.last_login_at is None
    assert stored.last_error
    assert "authorization" in stored.last_error.casefold()


# ============================================================================
# Phase 5.6 - session expiration, logout and shared session lock
# ============================================================================


class NavigationErrorPage(LoginPage):
    """Page whose navigation fails with a secret-bearing technical error."""

    async def goto(self, url: str, **kwargs) -> None:
        raise RuntimeError("navigation failed with token=SECRET-DO-NOT-PERSIST")


class LogoutContext(FakeContext):
    """Browser context that records the real Playwright cleanup path."""

    def __init__(self, page: FakePage):
        super().__init__(page)
        self.cookies_cleared = False
        self.storage_cleared = False
        self.storage_script = ""
        self.goto_urls: list[str] = []
        self.legacy_storage_calls = 0
        self.fail_cookies = False
        self.fail_storage = False
        self.forced_origin: str | None = None

        original_goto = page.goto

        async def tracked_goto(url: str, **kwargs) -> None:
            self.goto_urls.append(url)
            await original_goto(url, **kwargs)
            if self.forced_origin is not None:
                page.url = self.forced_origin

        page.goto = tracked_goto  # type: ignore[method-assign]
        page.evaluate = self._evaluate  # type: ignore[method-assign]

    async def clear_cookies(self) -> None:
        if self.fail_cookies:
            raise RuntimeError("cookies are locked")
        self.cookies_cleared = True

    async def clear_storage_state(self) -> None:
        # Playwright BrowserContext has no such method.  If production ever
        # calls it, this flag makes the regression visible in tests.
        self.legacy_storage_calls += 1

    async def _evaluate(self, script: str) -> None:
        if self.fail_storage:
            raise RuntimeError("storage is locked")
        self.storage_script = script
        self.storage_cleared = True


class FailingLogoutContext(LogoutContext):
    """Browser context whose cookie cleanup fails."""

    def __init__(self, page: FakePage):
        super().__init__(page)
        self.fail_cookies = True


class FailingStorageLogoutContext(LogoutContext):
    """Browser context whose DOM storage cleanup fails."""

    def __init__(self, page: FakePage):
        super().__init__(page)
        self.fail_storage = True


def _mark_account_authorized(
    database: Database, *, last_login_at: str | None = NOW.isoformat()
) -> None:
    """Put user 42's account into a previously-authorised state."""
    assert database.confirm_hh_account_authorized(
        42,
        last_checked_at=NOW.isoformat(),
        last_error="",
        last_login_at=last_login_at,
    )


def _session_client(
    settings,
    database: Database,
    account,
    page: FakePage,
    *,
    checker=None,
    lock: asyncio.Lock | None = None,
    context: FakeContext | None = None,
    now: datetime = NOW,
):
    user_settings = _approval_user_settings(settings)
    return HHClient(
        context or FakeContext(page),
        settings,
        database,
        ApprovalGuard(user_settings, database, now_factory=lambda: now),
        telegram_user_id=42,
        user_settings=user_settings,
        hh_account=account,
        sleep=lambda _seconds: asyncio.sleep(0),
        now_factory=lambda: now,
        session_lock=lock,
        session_checker=checker,
    )


def test_session_check_contradictory_page_signals_are_unknown(tmp_path: Path) -> None:
    """A login form plus an authenticated marker is not a definite logout."""
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    page = LoginPage(
        url=LOGIN_PAGE_URL,
        visible_selectors=(AUTHORIZED_MARKER, LOGIN_MARKER),
    )
    client = _session_client(settings, database, account_42, page)

    assert asyncio.run(client._authorization_state(page)) is LoginState.UNKNOWN


def test_authorized_url_with_login_control_is_unknown(tmp_path: Path) -> None:
    """An applicant URL plus a login control is contradictory, not logout."""
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    page = LoginPage(
        url=AUTHORIZED_PAGE_URL,
        visible_selectors=(LOGIN_MARKER,),
    )
    client = _session_client(settings, database, account_42, page)

    assert asyncio.run(client._authorization_state(page)) is LoginState.UNKNOWN


def test_check_session_persists_positive_result(tmp_path: Path) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    page = LoginPage(
        url=AUTHORIZED_PAGE_URL,
        visible_selectors=(AUTHORIZED_MARKER,),
    )
    client = _session_client(settings, database, account_42, page)

    assert asyncio.run(client.check_session()) is LoginState.AUTHORIZED

    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_AUTHORIZED
    assert stored.last_checked_at == NOW.isoformat()
    assert stored.last_login_at is None
    assert stored.last_error == ""


def test_positive_rechecks_do_not_rewrite_last_login_at(
    tmp_path: Path,
) -> None:
    """last_login_at records a real login, not every successful check."""
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    login_at = "2026-07-29T08:15:00+00:00"
    _mark_account_authorized(database, last_login_at=login_at)
    later = NOW + timedelta(hours=2)
    page = LoginPage(
        url=AUTHORIZED_PAGE_URL,
        visible_selectors=(AUTHORIZED_MARKER,),
    )
    client = _session_client(
        settings, database, account_42, page, now=later
    )

    assert asyncio.run(client.check_session()) is LoginState.AUTHORIZED
    assert asyncio.run(client.check_session()) is LoginState.AUTHORIZED

    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_AUTHORIZED
    assert stored.last_login_at == login_at
    assert stored.last_checked_at == later.isoformat()
    assert stored.last_error == ""


def test_check_session_technical_error_is_not_a_logout(tmp_path: Path) -> None:
    """CHECK_FAILED must not prove logout or expire a valid session."""
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database)
    client = _session_client(
        settings, database, account_42, NavigationErrorPage()
    )

    assert asyncio.run(client.check_session()) is LoginState.CHECK_FAILED

    stored = database.get_hh_account_model(42)
    assert stored is not None
    # A technical error keeps the prior-authorization state; it is neither a
    # logout nor an expiry.  The safe error still blocks the runtime gate.
    assert stored.status == HH_ACCOUNT_STATUS_AUTHORIZED
    assert "technical error" in stored.last_error.casefold()
    assert "SECRET" not in stored.last_error
    assert stored.last_login_at == NOW.isoformat()


def test_check_failure_keeps_expiry_memory_without_login_timestamp(
    tmp_path: Path,
) -> None:
    """Prior authorization is remembered even when no real login time exists."""
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database, last_login_at=None)
    client = _session_client(
        settings, database, account_42, NavigationErrorPage()
    )

    assert asyncio.run(client.check_session()) is LoginState.CHECK_FAILED
    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_AUTHORIZED
    assert stored.last_login_at is None

    login_page = LoginPage(
        url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,)
    )
    recheck = _session_client(settings, database, account_42, login_page)
    assert asyncio.run(recheck.check_session()) is LoginState.NOT_AUTHORIZED

    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_SESSION_EXPIRED
    assert stored.last_login_at is None


def test_previously_authorized_session_becomes_expired(tmp_path: Path) -> None:
    """NOT_AUTHORIZED after a confirmed login is a session expiry."""
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database)
    page = LoginPage(url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,))
    client = _session_client(settings, database, account_42, page)

    assert asyncio.run(client.check_session()) is LoginState.NOT_AUTHORIZED

    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_SESSION_EXPIRED
    assert "expired" in stored.last_error.casefold()


def test_never_authorized_session_becomes_login_required(tmp_path: Path) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    page = LoginPage(url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,))
    client = _session_client(settings, database, account_42, page)

    assert asyncio.run(client.check_session()) is LoginState.NOT_AUTHORIZED

    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_LOGIN_REQUIRED
    assert stored.last_login_at is None


def test_check_session_unknown_is_persisted(tmp_path: Path) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    client = _session_client(
        settings, database, account_42, FakePage()
    )

    assert asyncio.run(client.check_session()) is LoginState.UNKNOWN

    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_UNKNOWN
    assert "could not be verified" in stored.last_error.casefold()


def test_logout_confirmed_clears_session_and_persists_logged_out(
    tmp_path: Path,
) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database)
    page = LoginPage(url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,))
    context = LogoutContext(page)
    client = _session_client(
        settings, database, account_42, page, context=context
    )

    result = asyncio.run(client.logout())

    assert result.ok is True
    assert result.state is LoginState.NOT_AUTHORIZED
    assert result.reason == "confirmed"
    assert context.cookies_cleared is True
    assert context.storage_cleared is True
    assert "https://hh.kz/" in context.goto_urls
    assert context.storage_script == (
        "localStorage.clear(); sessionStorage.clear();"
    )
    assert context.legacy_storage_calls == 0
    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_LOGGED_OUT
    assert "closed" in stored.last_error.casefold()


def test_checks_after_explicit_logout_stay_logged_out(tmp_path: Path) -> None:
    """An inconclusive follow-up must not become a false expiry claim."""
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database)
    assert database.transition_hh_account(
        42,
        HH_ACCOUNT_STATUS_LOGGED_OUT,
        last_error="HH account session closed by user",
    )
    page = FakePage()
    client = _session_client(settings, database, account_42, page)

    assert asyncio.run(client.check_session()) is LoginState.UNKNOWN
    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_LOGGED_OUT
    assert "closed" in stored.last_error.casefold()

    page.url = LOGIN_PAGE_URL
    page.visible_selectors = {LOGIN_MARKER}
    assert asyncio.run(client.check_session()) is LoginState.NOT_AUTHORIZED
    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_LOGGED_OUT
    assert stored.last_login_at == NOW.isoformat()


def test_failed_logout_invalidation_keeps_explicit_logout_status(
    tmp_path: Path,
) -> None:
    """A failed repeat logout must not downgrade an explicit logout."""
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database)
    assert database.transition_hh_account(
        42,
        HH_ACCOUNT_STATUS_LOGGED_OUT,
        last_error="HH account session closed by user",
    )
    page = LoginPage(url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,))
    context = FailingLogoutContext(page)
    client = _session_client(
        settings, database, account_42, page, context=context
    )

    result = asyncio.run(client.logout())

    assert result.ok is False
    assert result.state is LoginState.CHECK_FAILED
    assert result.reason == "invalidate_failed"
    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_LOGGED_OUT
    assert stored.last_login_at == NOW.isoformat()


def test_logout_storage_cleanup_failure_is_not_a_success(
    tmp_path: Path,
) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database)
    page = LoginPage(url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,))
    context = FailingStorageLogoutContext(page)
    client = _session_client(
        settings, database, account_42, page, context=context
    )

    result = asyncio.run(client.logout())

    assert result.ok is False
    assert result.state is LoginState.CHECK_FAILED
    assert result.reason == "invalidate_failed"
    assert context.cookies_cleared is True
    assert context.storage_cleared is False
    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status != HH_ACCOUNT_STATUS_LOGGED_OUT


@pytest.mark.parametrize(
    "allowed_origin",
    [
        "https://hh.kz",
        "https://www.hh.kz",
        "https://hh.kz:443",
        "https://www.hh.kz:443",
    ],
)
def test_hh_origin_allows_only_exact_https_origins(allowed_origin: str) -> None:
    assert _is_hh_origin(allowed_origin) is True


@pytest.mark.parametrize(
    "foreign_origin",
    [
        "http://hh.kz",
        "https://hh.kz:444",
        "https://www.hh.kz:8443",
        "https://hh.kz.evil.example",
        "https://hh.kz@evil.example",
        "about:blank",
    ],
)
def test_hh_origin_rejects_foreign_and_lookalike_origins(
    foreign_origin: str,
) -> None:
    assert _is_hh_origin(foreign_origin) is False


@pytest.mark.parametrize(
    "malformed_url",
    ["https://hh.kz:not-a-port", "", "https://[::1"],
)
def test_hh_origin_returns_false_for_malformed_urls(malformed_url: str) -> None:
    assert _is_hh_origin(malformed_url) is False


@pytest.mark.parametrize(
    "foreign_origin",
    [
        "about:blank",
        "https://hh.kz.evil.example",
        "https://hh.kz@evil.example",
    ],
)
def test_logout_does_not_clear_storage_on_a_foreign_origin(
    tmp_path: Path, foreign_origin: str
) -> None:
    """localStorage.clear() must never run on a look-alike origin."""
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database)
    page = LoginPage(url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,))
    context = LogoutContext(page)
    context.forced_origin = foreign_origin
    client = _session_client(
        settings, database, account_42, page, context=context
    )

    result = asyncio.run(client.logout())

    assert result.ok is False
    assert result.reason == "invalidate_failed"
    assert context.cookies_cleared is True
    assert context.storage_cleared is False
    assert context.storage_script == ""
    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status != HH_ACCOUNT_STATUS_LOGGED_OUT


@pytest.mark.parametrize(
    "post_check_state",
    [LoginState.UNKNOWN, LoginState.CHECK_FAILED],
)
def test_logout_requires_positive_post_clear_confirmation(
    tmp_path: Path, post_check_state: LoginState
) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database)

    async def inconclusive_check() -> LoginState:
        return post_check_state

    page = LoginPage(url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,))
    context = LogoutContext(page)
    client = _session_client(
        settings,
        database,
        account_42,
        page,
        checker=inconclusive_check,
        context=context,
    )

    result = asyncio.run(client.logout())

    assert result.ok is False
    assert result.state is post_check_state
    assert result.reason == "unconfirmed"
    assert context.cookies_cleared is True
    assert context.storage_cleared is True
    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status != HH_ACCOUNT_STATUS_LOGGED_OUT


def test_logout_without_positive_confirmation_reports_failure(
    tmp_path: Path,
) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database)

    async def still_authorized() -> LoginState:
        return LoginState.AUTHORIZED

    page = LoginPage(url=AUTHORIZED_PAGE_URL)
    context = LogoutContext(page)
    client = _session_client(
        settings,
        database,
        account_42,
        page,
        checker=still_authorized,
        context=context,
    )

    result = asyncio.run(client.logout())

    assert result.ok is False
    assert result.state is LoginState.AUTHORIZED
    assert result.reason == "unconfirmed"
    stored = database.get_hh_account_model(42)
    assert stored is not None
    assert stored.status == HH_ACCOUNT_STATUS_AUTHORIZED


def test_logout_without_account_fails_safely(tmp_path: Path) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    page = LoginPage(url=LOGIN_PAGE_URL, visible_selectors=(LOGIN_MARKER,))
    context = LogoutContext(page)
    client = _session_client(settings, database, None, page, context=context)

    result = asyncio.run(client.logout())

    assert result.ok is False
    assert result.reason == "no_account"


def test_submit_application_requires_positive_session(tmp_path: Path) -> None:
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    user_settings = _approval_user_settings(settings)
    _setup_user_with_resume(database, 42, "Resume 42", NOW)
    permit = _setup_vacancy_and_approve(database, "job-blocked", 42, NOW)
    page = ResumeTrackingPage()
    client = _session_client(settings, database, None, page)

    async def failed_check() -> LoginState:
        return LoginState.CHECK_FAILED

    client.session_checker = failed_check
    permission = ApplicationPermission("job-blocked", permit, 42)

    assert asyncio.run(client.submit_application(permission)) is False
    assert page.actions == []
    assert database.get("job-blocked").status is VacancyStatus.APPROVED


def test_check_logout_and_submit_share_one_session_lock(tmp_path: Path) -> None:
    """No check may overlap with a logout or a physical submit."""
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database)
    user_settings = _approval_user_settings(settings)
    _setup_user_with_resume(database, 42, "Resume 42", NOW)
    permit = _setup_vacancy_and_approve(database, "job-locked", 42, NOW)

    active = 0
    peak = 0

    async def slow_unauthorized_check() -> LoginState:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return LoginState.NOT_AUTHORIZED

    shared_lock = asyncio.Lock()
    page = ResumeTrackingPage()
    context = LogoutContext(page)
    client = _session_client(
        settings,
        database,
        account_42,
        page,
        checker=slow_unauthorized_check,
        lock=shared_lock,
        context=context,
    )
    assert client.session_lock is shared_lock

    permission = ApplicationPermission("job-locked", permit, 42)

    async def scenario() -> None:
        await asyncio.gather(
            client.check_session(),
            client.logout(),
            client.submit_application(permission),
        )

    asyncio.run(scenario())

    assert peak == 1


def test_submit_holds_session_lock_through_physical_click(tmp_path: Path) -> None:
    """The lock must survive the session check and reach the actual send."""
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    account_42, _ = _setup_hh_accounts(database)
    _mark_account_authorized(database)
    user_settings = _approval_user_settings(settings)
    _setup_user_with_resume(database, 42, "Resume FortyTwo", NOW)
    permit = _setup_vacancy_and_approve(database, "job-locked-submit", 42, NOW)

    shared_lock = asyncio.Lock()

    async def authorized_check() -> LoginState:
        return LoginState.AUTHORIZED

    class LockObservingPage(ResumeTrackingPage):
        def __init__(self) -> None:
            super().__init__(expected_resume="Resume FortyTwo")
            self.lock_held_at_submit: bool | None = None

        def _on_click(self, selector_type: str = "") -> None:
            if selector_type == "submit":
                self.lock_held_at_submit = shared_lock.locked()
            super()._on_click(selector_type)

    page = LockObservingPage()
    client = _session_client(
        settings,
        database,
        account_42,
        page,
        checker=authorized_check,
        lock=shared_lock,
    )
    permission = ApplicationPermission("job-locked-submit", permit, 42)

    assert asyncio.run(client.submit_application(permission)) is True
    assert page.lock_held_at_submit is True
    assert shared_lock.locked() is False


def test_check_session_without_account_cannot_return_authorized(
    tmp_path: Path,
) -> None:
    """The session_checker seam must not bypass the HH account gate."""
    settings = _login_settings(tmp_path)
    database = Database(settings.database_path)
    database.init()
    user_settings = _approval_user_settings(settings)

    calls: list[str] = []

    async def authorized_checker() -> LoginState:
        calls.append("called")
        return LoginState.AUTHORIZED

    client = HHClient(
        FakeContext(LoginPage(url=LOGIN_PAGE_URL)),
        settings,
        database,
        ApprovalGuard(user_settings, database, now_factory=lambda: NOW),
        telegram_user_id=42,
        user_settings=user_settings,
        sleep=lambda _seconds: asyncio.sleep(0),
        now_factory=lambda: NOW,
        session_checker=authorized_checker,
    )
    assert client.hh_account is None

    state = asyncio.run(client.check_session())

    assert state is LoginState.CHECK_FAILED
    # The checker must never run without a verified account.
    assert calls == []
    # Nothing may be persisted: there is no account row at all, so no
    # AUTHORIZED (or any other) status can have been written.
    assert database.get_hh_account(42) is None
    assert database.get_hh_account_model(42) is None
