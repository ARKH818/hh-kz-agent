import asyncio
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from browser_backend import (
    BrowserLaunchError,
    CloakBrowserBackend,
    PlaywrightBrowserBackend,
    create_browser_backend,
)
from config import load_settings
from tests.test_config import VALID_ENV, write_profile


class FakeContext:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def settings(tmp_path: Path, backend: str = "cloakbrowser"):
    loaded = load_settings(
        profile_path=write_profile(tmp_path),
        environ={**VALID_ENV, "BROWSER_BACKEND": backend},
    )
    return replace(loaded, browser_profile_dir=tmp_path / "profile")


def test_factory_selects_configured_backend(tmp_path: Path) -> None:
    assert isinstance(
        create_browser_backend(settings(tmp_path, "cloakbrowser")),
        CloakBrowserBackend,
    )
    assert isinstance(
        create_browser_backend(settings(tmp_path, "playwright")),
        PlaywrightBrowserBackend,
    )


def test_factory_rejects_unknown_backend(tmp_path: Path) -> None:
    invalid = replace(settings(tmp_path), browser_backend="unknown")

    with pytest.raises(ValueError, match="Unknown browser backend: unknown"):
        create_browser_backend(invalid)


@pytest.mark.parametrize(
    ("backend_name", "backend_type"),
    [
        ("cloakbrowser", CloakBrowserBackend),
        ("playwright", PlaywrightBrowserBackend),
    ],
)
def test_factory_uses_explicit_profile_dir(
    tmp_path: Path, backend_name: str, backend_type: type
) -> None:
    """Phase 5.3: an explicit per-account profile directory is used as-is."""
    explicit_profile = tmp_path / "accounts" / "hh-account-7"

    backend = create_browser_backend(
        settings(tmp_path, backend_name),
        profile_dir=explicit_profile,
    )

    assert isinstance(backend, backend_type)
    assert backend.profile_dir == explicit_profile


@pytest.mark.parametrize(
    ("backend_name", "backend_type"),
    [
        ("cloakbrowser", CloakBrowserBackend),
        ("playwright", PlaywrightBrowserBackend),
    ],
)
def test_factory_without_profile_dir_preserves_settings_default(
    tmp_path: Path, backend_name: str, backend_type: type
) -> None:
    """Phase 5.3: existing create_browser_backend(settings) callers are unchanged."""
    app_settings = settings(tmp_path, backend_name)

    backend = create_browser_backend(app_settings)

    assert isinstance(backend, backend_type)
    assert backend.profile_dir == app_settings.browser_profile_dir


@pytest.mark.parametrize(
    ("backend_name", "backend_type"),
    [
        ("cloakbrowser", CloakBrowserBackend),
        ("playwright", PlaywrightBrowserBackend),
    ],
)
def test_persistent_profile_and_headless_are_passed_to_launcher(
    tmp_path: Path, backend_name: str, backend_type: type
) -> None:
    calls: list[tuple[Path, bool]] = []
    context = FakeContext()

    async def launcher(profile_dir: Path, *, headless: bool):
        calls.append((Path(profile_dir), headless))
        return context

    backend = backend_type(tmp_path / "persistent", False, launcher=launcher)
    started = asyncio.run(backend.start())
    asyncio.run(backend.close())

    assert started is context
    assert calls == [(tmp_path / "persistent", False)]
    assert context.closed


@pytest.mark.parametrize(
    ("backend_name", "backend_type"),
    [
        ("cloakbrowser", CloakBrowserBackend),
        ("playwright", PlaywrightBrowserBackend),
    ],
)
def test_repeated_start_reuses_profile_without_relaunch(
    tmp_path: Path, backend_name: str, backend_type: type
) -> None:
    """Phase 5.5: start() is idempotent and never creates a second profile."""
    calls: list[tuple[Path, bool]] = []
    context = FakeContext()

    async def launcher(profile_dir: Path, *, headless: bool):
        calls.append((Path(profile_dir), headless))
        return context

    profile = tmp_path / "account-profile"
    backend = backend_type(profile, False, launcher=launcher)

    first = asyncio.run(backend.start())
    second = asyncio.run(backend.start())

    assert first is context
    assert second is context
    assert calls == [(profile, False)]


@pytest.mark.parametrize(
    ("backend_name", "backend_type"),
    [
        ("cloakbrowser", CloakBrowserBackend),
        ("playwright", PlaywrightBrowserBackend),
    ],
)
def test_restart_reuses_same_profile_and_preserves_its_contents(
    tmp_path: Path, backend_name: str, backend_type: type
) -> None:
    """Phase 5.5: a restart uses the same directory and deletes nothing."""
    calls: list[tuple[Path, bool]] = []
    contexts = [FakeContext(), FakeContext()]

    async def launcher(profile_dir: Path, *, headless: bool):
        calls.append((Path(profile_dir), headless))
        return contexts[len(calls) - 1]

    profile = tmp_path / "account-profile"
    profile.mkdir()
    marker = profile / "cookies.marker"
    marker.write_text("persisted-session", encoding="utf-8")

    backend = backend_type(profile, True, launcher=launcher)
    asyncio.run(backend.start())
    asyncio.run(backend.close())
    asyncio.run(backend.start())
    asyncio.run(backend.close())

    assert calls == [(profile, True), (profile, True)]
    assert marker.read_text(encoding="utf-8") == "persisted-session"
    assert contexts[0].closed is True
    assert contexts[1].closed is True


def test_cloakbrowser_launch_error_is_actionable(tmp_path: Path) -> None:
    async def broken_launcher(profile_dir: Path, *, headless: bool):
        raise RuntimeError("binary blocked")

    backend = CloakBrowserBackend(tmp_path / "profile", False, launcher=broken_launcher)

    with pytest.raises(BrowserLaunchError) as raised:
        asyncio.run(backend.start())

    message = str(raised.value)
    assert "CloakBrowser failed to start" in message
    assert "BROWSER_BACKEND=playwright" in message
    assert "binary blocked" in message


def test_browser_backends_do_not_receive_approval_state(tmp_path: Path) -> None:
    backends = [
        CloakBrowserBackend(tmp_path / "cloak", False, launcher=None),
        PlaywrightBrowserBackend(tmp_path / "playwright", False, launcher=None),
    ]

    assert all(
        not any("approval" in attribute or "database" in attribute for attribute in vars(backend))
        for backend in backends
    )


def test_browser_profiles_and_storage_state_are_ignored_by_git() -> None:
    assert all(
        subprocess.run(
            ["git", "check-ignore", "--quiet", path], check=False
        ).returncode
        == 0
        for path in (
            ".browser-profile/session",
            ".browser-profile.backup/session",
            "state.json",
            "storage_state.json",
            "backups/agent-backup.db",
        )
    )