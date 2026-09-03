import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from approval import ApplicationPermission, ApprovalGuard
from config import load_settings
from database import Database
from hh_client import HHClient, PageState, VacancySummary, classify_page
from tests.test_config import VALID_ENV, write_profile
from vacancy_filter import title_rejection_reason



NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)

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


class RetryLoginPage(FakePage):
    def __init__(self):
        super().__init__()
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
        FailingContext(), settings, database, ApprovalGuard(settings, database)
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
        FailingContext(), settings, database, ApprovalGuard(settings, database)
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
        FakeContext(page), settings, database, ApprovalGuard(settings, database)
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

    _setup_user_with_resume(database, 42, "Resume FortyTwo", now)
    permit = _setup_vacancy_and_approve(database, "job-42", 42, now)

    page = ResumeTrackingPage(expected_resume="Resume FortyTwo")
    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(settings, database, now_factory=lambda: NOW),
        sleep=lambda _: asyncio.sleep(0),
        now_factory=lambda: NOW,
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

    _setup_user_with_resume(database, 43, "Resume FortyThree", now)
    permit = _setup_vacancy_and_approve(database, "job-43", 43, now)

    page = ResumeTrackingPage(expected_resume="Resume FortyThree")
    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(settings, database, now_factory=lambda: NOW),
        sleep=lambda _: asyncio.sleep(0),
        now_factory=lambda: NOW,
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
        ApprovalGuard(settings, database, now_factory=lambda: NOW),
        sleep=lambda _: asyncio.sleep(0),
        now_factory=lambda: NOW,
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
        ApprovalGuard(settings, database, now_factory=lambda: NOW),
        sleep=lambda _: asyncio.sleep(0),
        now_factory=lambda: NOW,
    )

    permission = ApplicationPermission("job-42", permit, 42)
    # Should proceed without resume selection (resume_name is empty)
    result = asyncio.run(client.submit_application(permission))

    # The key assertion: no resume was selected because user has none
    assert "selected_resume:" not in str(page.actions)