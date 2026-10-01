from pathlib import Path

import pytest
import yaml

from config import ConfigError, UserSettings, load_settings, load_user_settings


VALID_ENV = {
    "TG_BOT_TOKEN": "test-token",
    "ENABLE_REAL_APPLY": "false",
    "BROWSER_HEADLESS": "false",
    "CHECK_INTERVAL_MINUTES": "30",
    "MAX_APPLICATIONS_PER_DAY": "5",
    "MAX_VACANCIES_PER_QUERY": "20",
    "MAX_PAGES_PER_QUERY": "2",
    "MIN_SECONDS_BETWEEN_ACTIONS": "5",
}

VALID_PROFILE = {
    "candidate": {
        "name": "Test Candidate",
        "location": "Moscow",
        "desired_positions": ["Python developer"],
        "experience_summary": "Built internal Python services.",
        "education": "",
        "technologies": ["Python", "SQLite"],
        "projects": [],
        "github_url": "",
        "salary_expectation": "",
        "work_format": ["remote"],
        "excluded_positions": ["sales"],
        "additional_information": "",
    },
    "hh": {
        "resume_name": "Python developer",
        "search_queries": ["Python developer"],
        "areas": ["1"],
        "experience_filters": ["between1And3"],
    },
}


def write_profile(tmp_path: Path, data: dict | None = None) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text(
        yaml.safe_dump(data or VALID_PROFILE, allow_unicode=True), encoding="utf-8"
    )
    return path


def test_load_settings_uses_safe_defaults(tmp_path: Path) -> None:
    settings = load_settings(
        profile_path=write_profile(tmp_path), environ=VALID_ENV
    )

    assert settings.app_mode == "dry_run"
    assert settings.enable_real_apply is False
    assert settings.browser_backend == "cloakbrowser"
    assert settings.browser_headless is False
    assert settings.browser_profile_dir.name == ".browser-profile"
    assert settings.profile.candidate.name == "Test Candidate"
    assert settings.profile.hh.search_queries == ("Python developer",)
    assert settings.circuit_breaker_min_sample == 5
    assert settings.circuit_breaker_unknown_ratio == 0.8
    assert settings.circuit_breaker_page_errors == 3


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("APP_MODE", "automatic", "APP_MODE must be dry_run or approval"),
        (
            "BROWSER_BACKEND",
            "unknown",
            "BROWSER_BACKEND must be cloakbrowser or playwright",
        ),
        (
            "ENABLE_REAL_APPLY",
            "yes",
            "ENABLE_REAL_APPLY must be true or false",
        ),
        (
            "MAX_APPLICATIONS_PER_DAY",
            "0",
            "MAX_APPLICATIONS_PER_DAY must be a positive integer",
        ),
    ],
)
def test_load_settings_rejects_invalid_environment(
    tmp_path: Path, key: str, value: str, message: str
) -> None:
    environ = {**VALID_ENV, key: value}

    with pytest.raises(ConfigError, match=message):
        load_settings(profile_path=write_profile(tmp_path), environ=environ)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            "CIRCUIT_BREAKER_MIN_SAMPLE",
            "0",
            "CIRCUIT_BREAKER_MIN_SAMPLE must be a positive integer",
        ),
        (
            "CIRCUIT_BREAKER_UNKNOWN_RATIO",
            "0",
            "CIRCUIT_BREAKER_UNKNOWN_RATIO must be greater than 0 and at most 1",
        ),
        (
            "CIRCUIT_BREAKER_UNKNOWN_RATIO",
            "1.1",
            "CIRCUIT_BREAKER_UNKNOWN_RATIO must be greater than 0 and at most 1",
        ),
        (
            "CIRCUIT_BREAKER_PAGE_ERRORS",
            "no",
            "CIRCUIT_BREAKER_PAGE_ERRORS must be an integer",
        ),
    ],
)
def test_load_settings_rejects_invalid_circuit_breaker_values(
    tmp_path: Path, key: str, value: str, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_settings(
            profile_path=write_profile(tmp_path),
            environ={**VALID_ENV, key: value},
        )


def test_load_settings_reports_missing_secret_without_traceback_text(
    tmp_path: Path,
) -> None:
    environ = {**VALID_ENV, "TG_BOT_TOKEN": ""}

    with pytest.raises(ConfigError) as raised:
        load_settings(profile_path=write_profile(tmp_path), environ=environ)

    assert str(raised.value).startswith("Configuration error:")
    assert "TG_BOT_TOKEN is required" in str(raised.value)
    assert "Traceback" not in str(raised.value)


def test_load_settings_aggregates_required_profile_errors(tmp_path: Path) -> None:
    profile = {
        **VALID_PROFILE,
        "candidate": {
            **VALID_PROFILE["candidate"],
            "name": "",
            "desired_positions": [],
            "experience_summary": "",
        },
        "hh": {**VALID_PROFILE["hh"], "resume_name": "", "search_queries": []},
    }

    with pytest.raises(ConfigError) as raised:
        load_settings(profile_path=write_profile(tmp_path, profile), environ=VALID_ENV)

    message = str(raised.value)
    assert "candidate.name is required" in message
    assert "candidate.desired_positions must not be empty" in message
    assert "candidate.experience_summary is required" in message
    assert "hh.resume_name is required" in message
    assert "hh.search_queries must not be empty" in message


def test_load_settings_rejects_missing_profile_file_when_explicitly_requested(tmp_path: Path) -> None:
    """If user explicitly passes a non-existent profile path, it's an error."""
    with pytest.raises(ConfigError, match="profile file not found"):
        load_settings(profile_path=tmp_path / "missing.yaml", environ=VALID_ENV)


def test_load_settings_works_without_profile_file(tmp_path: Path) -> None:
    """App should start when profile.yaml is completely absent."""
    settings = load_settings(profile_path=None, environ=VALID_ENV)

    assert settings.tg_bot_token == "test-token"
    assert settings.app_mode == "dry_run"
    # Profile should be empty defaults
    assert settings.profile.candidate.name == ""
    assert settings.profile.candidate.desired_positions == ()
    assert settings.profile.hh.resume_name == ""
    assert settings.profile.hh.search_queries == ()


def test_load_settings_uses_env_profile_path_when_present(tmp_path: Path) -> None:
    """PROFILE_PATH env var should still work when set."""
    profile_path = write_profile(tmp_path)
    environ = {**VALID_ENV, "PROFILE_PATH": str(profile_path)}
    settings = load_settings(profile_path=None, environ=environ)

    assert settings.profile.candidate.name == "Test Candidate"
    assert settings.profile.hh.resume_name == "Python developer"


def test_load_settings_empty_profile_does_not_break_startup(tmp_path: Path) -> None:
    """Empty profile should not raise ConfigError for missing fields."""
    # No profile file, no PROFILE_PATH env var
    settings = load_settings(profile_path=None, environ=VALID_ENV)

    # Should not raise - empty profile is valid when not explicitly loaded
    assert settings.profile.candidate.name == ""
    assert settings.profile.hh.resume_name == ""


def test_load_settings_works_without_tg_user_id(tmp_path: Path) -> None:
    """App should start when TG_USER_ID is not set."""
    settings = load_settings(
        profile_path=write_profile(tmp_path),
        environ=VALID_ENV,
    )

    assert settings.tg_bot_token == "test-token"
    assert settings.app_mode == "dry_run"
    assert settings.browser_backend == "cloakbrowser"

# ============================================================================
# Stage 4\6 — UserSettings / per-user configuration
# ============================================================================

def test_settings_carries_user_defaults_snapshot(tmp_path: Path) -> None:
    """Global Settings holds an immutable default UserSettings snapshot."""
    settings = load_settings(
        profile_path=write_profile(tmp_path), environ=VALID_ENV
    )
    assert isinstance(settings.user_defaults, UserSettings)
    assert settings.user_defaults.app_mode == "dry_run"
    assert settings.user_defaults.profile.candidate.name == "Test Candidate"


def test_load_user_settings_returns_isolated_instances(tmp_path: Path) -> None:
    """Two calls to load_user_settings return independent objects."""
    settings = load_settings(
        profile_path=write_profile(tmp_path), environ=VALID_ENV
    )
    a = load_user_settings(settings)
    b = load_user_settings(settings)
    assert a is not b
    assert a == b  # same values, different objects


def test_user_settings_override_does_not_affect_defaults(tmp_path: Path) -> None:
    """Overriding a field on one UserSettings does not change the snapshot."""
    settings = load_settings(
        profile_path=write_profile(tmp_path), environ=VALID_ENV
    )
    custom = load_user_settings(settings, app_mode="approval")
    assert custom.app_mode == "approval"
    assert settings.user_defaults.app_mode == "dry_run"


def test_global_settings_keeps_all_existing_fields(tmp_path: Path) -> None:
    """Settings retains every pre-4\6 field (no regression)."""
    settings = load_settings(
        profile_path=write_profile(tmp_path), environ=VALID_ENV
    )
    for field_name in (
        "tg_bot_token", "app_mode", "enable_real_apply", "browser_backend",
        "browser_headless", "browser_profile_dir", "database_path", "log_path",
        "check_interval_minutes", "max_applications_per_day",
        "max_vacancies_per_query", "max_pages_per_query",
        "min_seconds_between_actions", "approval_ttl_minutes",
        "captcha_timeout_seconds", "captcha_max_attempts",
        "circuit_breaker_min_sample", "circuit_breaker_unknown_ratio",
        "circuit_breaker_page_errors", "profile",
    ):
        assert hasattr(settings, field_name), field_name


def test_user_settings_is_frozen(tmp_path: Path) -> None:
    """UserSettings is immutable; mutation raises FrozenInstanceError."""
    import dataclasses

    settings = load_settings(
        profile_path=write_profile(tmp_path), environ=VALID_ENV
    )
    us = load_user_settings(settings)
    with pytest.raises(dataclasses.FrozenInstanceError):
        us.app_mode = "approval"  # type: ignore[misc]