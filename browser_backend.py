from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

from config import Settings


Launcher = Callable[..., Awaitable[Any]]


class BrowserLaunchError(RuntimeError):
    pass


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
            self.profile_dir.mkdir(parents=True, exist_ok=True)

            if self._launcher is None:
                from cloakbrowser import launch_persistent_context_async

                launcher = launch_persistent_context_async
            else:
                launcher = self._launcher

            context = await launcher(
                str(self.profile_dir),
                headless=self.headless,
            )

            if context is None:
                raise RuntimeError("CloakBrowser returned an empty context")

            self._context = context

            return self._context

        except Exception as exc:
            self._context = None
            raise BrowserLaunchError(
                "CloakBrowser failed to start: "
                f"{exc}. "
                "Set BROWSER_BACKEND=playwright to use the Playwright backend."
            ) from exc

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
            self.profile_dir.mkdir(parents=True, exist_ok=True)

            launcher = self._launcher or self._launch

            self._context = await launcher(
                self.profile_dir,
                headless=self.headless,
            )

            if self._context is None:
                raise RuntimeError("Playwright returned an empty context")

            return self._context

        except Exception as exc:
            raise BrowserLaunchError(
                f"Playwright failed to start: {exc}"
            ) from exc

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


def create_browser_backend(settings: Settings) -> BrowserBackend:
    if settings.browser_backend == "cloakbrowser":
        return CloakBrowserBackend(
            settings.browser_profile_dir,
            settings.browser_headless,
        )

    if settings.browser_backend == "playwright":
        return PlaywrightBrowserBackend(
            settings.browser_profile_dir,
            settings.browser_headless,
        )

    raise ValueError(
        f"Unknown browser backend: {settings.browser_backend}"
    )