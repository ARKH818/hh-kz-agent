from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

from config import Settings
from logging_setup import safe_exception_message
from security_utils import ensure_validated_directory, protect_browser_profile


Launcher = Callable[..., Awaitable[Any]]


class BrowserLaunchError(RuntimeError):
    pass


def _launch_error(backend: str, exc: BaseException) -> BrowserLaunchError:
    """Build a launch error without copying a raw browser exception."""
    detail = safe_exception_message(exc, operation=f"{backend}_start")
    hint = (
        " Set BROWSER_BACKEND=playwright to use the Playwright backend."
        if backend == "CloakBrowser"
        else ""
    )
    return BrowserLaunchError(f"{backend} failed to start: {detail}.{hint}")


class BrowserBackend(Protocol):
    async def start(self) -> Any:
        ...

    async def close(self) -> None:
        ...


class CloakBrowserBackend:
    def __init__(
        self,
        profile_dir: Path,
        headless: bool,
        launcher: Launcher | None = None,
    ):
        self.profile_dir = profile_dir
        self.headless = headless
        self._launcher = launcher
        self._context: Any = None

    async def start(self) -> Any:
        if self._context is not None:
            return self._context

        try:
            # Each component is validated before it is created, and the final
            # path is checked again before the browser is launched.
            profile_dir = ensure_validated_directory(
                self.profile_dir, kind="browser profile"
            )
            protect_browser_profile(profile_dir)

            if self._launcher is None:
                from cloakbrowser import launch_persistent_context_async

                launcher = launch_persistent_context_async
            else:
                launcher = self._launcher

            context = await launcher(
                str(profile_dir),
                headless=self.headless,
            )

            if context is None:
                raise RuntimeError("CloakBrowser returned an empty context")

            self._context = context

            return self._context

        except Exception as exc:
            self._context = None
            raise _launch_error("CloakBrowser", exc) from exc

    async def close(self) -> None:
        context = self._context
        self._context = None

        if context is not None:
            try:
                await context.close()
            except Exception:
                pass


class PlaywrightBrowserBackend:
    def __init__(
        self,
        profile_dir: Path,
        headless: bool,
        launcher: Launcher | None = None,
    ):
        self.profile_dir = profile_dir
        self.headless = headless
        self._launcher = launcher
        self._context: Any = None
        self._playwright: Any = None

    async def _launch(
        self,
        profile_dir: Path,
        *,
        headless: bool,
    ) -> Any:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()

        try:
            return await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=headless,
            )
        except Exception:
            await self._playwright.stop()
            self._playwright = None
            raise

    async def start(self) -> Any:
        if self._context is not None:
            return self._context

        try:
            # Each component is validated before it is created, and the final
            # path is checked again before the browser is launched.
            profile_dir = ensure_validated_directory(
                self.profile_dir, kind="browser profile"
            )
            protect_browser_profile(profile_dir)

            launcher = self._launcher or self._launch

            self._context = await launcher(
                profile_dir,
                headless=self.headless,
            )

            if self._context is None:
                raise RuntimeError("Playwright returned an empty context")

            return self._context

        except Exception as exc:
            raise _launch_error("Playwright", exc) from exc

    async def close(self) -> None:
        context = self._context
        playwright = self._playwright

        self._context = None
        self._playwright = None

        if context is not None:
            try:
                await context.close()
            except Exception:
                pass

        if playwright is not None:
            try:
                await playwright.stop()
            except Exception:
                pass


def create_browser_backend(
    settings: Settings,
    profile_dir: Path | str | None = None,
) -> BrowserBackend:
    """Create the configured persistent-context browser backend.

    Phase 5.3: an explicit ``profile_dir`` lets the caller select the
    persistent profile owned by a specific HHAccount.  When it is not
    supplied the previous behaviour (``settings.browser_profile_dir``)
    is preserved for every existing caller.
    """
    selected_profile_dir = (
        Path(profile_dir)
        if profile_dir is not None
        else settings.browser_profile_dir
    )

    if settings.browser_backend == "cloakbrowser":
        return CloakBrowserBackend(
            selected_profile_dir,
            settings.browser_headless,
        )

    if settings.browser_backend == "playwright":
        return PlaywrightBrowserBackend(
            selected_profile_dir,
            settings.browser_headless,
        )

    raise ValueError(
        f"Unknown browser backend: {settings.browser_backend}"
    )