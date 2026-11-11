from __future__ import annotations

import asyncio
import html
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InputRichBlockDetails,
    InputRichBlockDivider,
    InputRichBlockParagraph,
    InputRichBlockSectionHeading,
    InputRichMessage,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    RichTextBold,
    RichTextUrl,
)

from approval import ApprovalService
from config import Settings
from database import Database, Vacancy
from runtime_context import UserRuntimeContext
from updater import ReleaseInfo, check_github_release
from version import __version__


logger = logging.getLogger(__name__)
PRIVATE_REPLY = "This bot is private."
NO_RUNTIME_REPLY = "Agent runtime is not running for your account."


class TelegramService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        approval_service: ApprovalService,
        contexts: dict[int, UserRuntimeContext] | None = None,
        *,
        bot: Any | None = None,
        dispatcher: Dispatcher | None = None,
        now_factory: Callable[[], datetime] | None = None,
    ):
        self.settings = settings
        self.database = database
        self.approval_service = approval_service
        self._contexts: dict[int, UserRuntimeContext] = (
            contexts if contexts is not None else {}
        )
        self.bot = bot or Bot(token=settings.tg_bot_token)
        self.dispatcher = dispatcher or Dispatcher()
        self.now_factory = now_factory or (lambda: datetime.now().astimezone())
        self._captcha_futures: dict[int, asyncio.Future[str | None]] = {}
        self.latest_release: ReleaseInfo | None = None
        self._notified_release_tag: str | None = None
        self._register_handlers()

    def authorized(self, user_id: int) -> bool:
        user = self.database.get_user(user_id)
        return user is not None and user["is_active"] == 1

    def _runtime_context(self, user_id: int) -> UserRuntimeContext | None:
        return self._contexts.get(user_id)

    def command(
        self,
        name: str,
        user_id: int,
        *,
        username: str | None = None,
        first_name: str | None = None,
    ) -> str:
        if name == "start":
            self.database.get_or_create_user(
                telegram_user_id=user_id,
                username=username,
                first_name=first_name,
                now=self.now_factory(),
            )
            if self.authorized(user_id):
                return "Welcome back! HH.kz Agent is ready."
            return "Welcome! You are now registered with HH.kz Agent."
        if not self.authorized(user_id):
            return PRIVATE_REPLY
        if name == "pause":
            context = self._runtime_context(user_id)
            if context is None:
                return NO_RUNTIME_REPLY
            context.paused = True
            context.next_run_at = None
            logger.info("agent_paused user=%s", user_id)
            return "Agent paused."
        if name == "resume":
            context = self._runtime_context(user_id)
            if context is None:
                return NO_RUNTIME_REPLY
            context.paused = False
            context.circuit_reason = ""
            context.consecutive_search_errors = 0
            context.next_run_at = None
            context.wake_event.set()
            logger.info("agent_resumed user=%s", user_id)
            return "Agent resumed."
        if name == "status":
            context = self._runtime_context(user_id)
            if context is None:
                return NO_RUNTIME_REPLY
            stats = self.database.stats(telegram_user_id=user_id)
            processed = sum(stats.values())
            version_str = f"v{__version__}"
            if self.latest_release is not None:
                version_str += f" (доступно обновление {self.latest_release.tag_name})"
            # Stage 4\6: the mode shown is the calling user's own setting.
            user_mode = (
                context.user_settings.app_mode
                if context is not None
                else self.settings.user_defaults.app_mode
            )
            return (
                "версия: " + version_str + "\n"
                "mode: " + user_mode + "\n"
                "state: " + ("paused" if context.paused else "running") + "\n"
                "processed: " + str(processed) + "\n"
                "applied today: " + str(self.database.applied_today(self.now_factory(), telegram_user_id=user_id))
            )
        if name == "pending":
            pending = self.database.pending(telegram_user_id=user_id)
            return (
                "No pending vacancies."
                if not pending
                else "\n".join(f"{item.id}: {item.title}" for item in pending)
            )
        if name == "stats":
            return "\n".join(
                f"{status}: {count}"
                for status, count in self.database.stats(telegram_user_id=user_id).items()
                if count
            ) or "No vacancies recorded."
        if name == "diagnostics":
            return self._diagnostics(user_id)
        if name == "cancel":
            future = self._captcha_futures.get(user_id)
            if future is not None and not future.done():
                future.set_result(None)
            return "Current input request cancelled."
        return "Unknown command."

    def _diagnostics(self, user_id: int) -> str:
        context = self._runtime_context(user_id)
        version_str = f"v{__version__}"
        if self.latest_release is not None:
            version_str += f" (доступно: {self.latest_release.tag_name})"
        if context is None:
            return (
                "версия: " + version_str + "\n"
                "runtime not running"
            )
        run = self.database.latest_search_run(telegram_user_id=user_id)
        if run is None:
            return "версия: " + version_str + "\nNo search diagnostics recorded."
        started = datetime.fromisoformat(run.started_at)
        finished = datetime.fromisoformat(run.finished_at)
        reasons = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(run.rejection_reasons.items())
        ) or "нет"
        errors = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(run.error_reasons.items())
        ) or "нет"
        breaker = (
            f"open ({context.circuit_reason})"
            if context.circuit_reason
            else "closed"
        )
        if context.paused:
            next_run = "paused"
        elif context.next_run_at is None:
            next_run = "после текущего цикла"
        else:
            next_run = context.next_run_at.astimezone().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        return (
            "версия: " + version_str + "\n"
            "последний цикл: " + started.astimezone().strftime("%Y-%m-%d %H:%M:%S") + "\n"
            "состояние: " + run.state + "\n"
            "длительность: " + str(int((finished - started).total_seconds())) + " с\n"
            "запросы: " + str(run.query_count) + "\n"
            "найдено: " + str(run.found_results) + "\n"
            "новые: " + str(run.new_vacancies) + "\n"
            "дубли: " + str(run.duplicates) + "\n"
            "отклонено — фильтр: " + str(run.rejected_by_filter) + "\n"
            "причины: " + reasons + "\n"
            "карточки: " + str(run.telegram_cards) + "\n"
            "ошибки: " + str(run.error_count) + " (" + errors + ")\n"
            "последняя ошибка: " + (run.last_safe_error or "нет") + "\n"
            "breaker: " + breaker + "\n"
            "last breaker: " + (run.circuit_reason or "none") + "\n"
            "следующий запуск: " + next_run
        )
    def _register_handlers(self) -> None:
        for name in (
            "start",
            "status",
            "pause",
            "resume",
            "pending",
            "stats",
            "diagnostics",
            "cancel",
        ):
            self.dispatcher.message.register(self._command_handler, Command(name))
        self.dispatcher.callback_query.register(
            self._callback_handler,
            F.data.startswith("apply:") | F.data.startswith("skip:"),
        )
        self.dispatcher.message.register(self._text_handler)

    async def _command_handler(self, message: Message) -> None:
        name = (message.text or "").split()[0].lstrip("/").split("@")[0]
        user_id = message.from_user.id
        # /start is allowed for unauthorized users to enable registration
        if name == "start":
            username = message.from_user.username
            first_name = message.from_user.first_name
            await message.answer(
                self.command(name, user_id, username=username, first_name=first_name)
            )
            return
        if not self.authorized(user_id):
            await message.answer(PRIVATE_REPLY)
            return
        await message.answer(self.command(name, user_id))

    async def _callback_handler(self, callback: CallbackQuery) -> None:
        if not self.authorized(callback.from_user.id):
            await self._answer_callback(callback, PRIVATE_REPLY, show_alert=True)
            return
        action, separator, job_id = (callback.data or "").partition(":")
        if len((callback.data or "").encode()) > 64:
            await self._answer_callback(
                callback, "Некорректное действие.", show_alert=True
            )
            return
        if not separator or action not in {"apply", "skip"} or not job_id:
            await self._answer_callback(
                callback, "Некорректное действие.", show_alert=True
            )
            return
        if action == "apply":
            await callback.answer("⏳ Отправляем отклик...", show_alert=False)
            result = await self.approval_service.approve_and_apply(
                job_id, callback.from_user.id
            )
            vacancy = self.database.get(job_id, telegram_user_id=callback.from_user.id)
            if result.ok:
                await self.notify("✓ Отклик отправлен", callback.from_user.id)
                if callback.message:
                    await callback.message.edit_reply_markup(reply_markup=None)
            else:
                if vacancy and vacancy.error_text == "questionnaire_required":
                    await self.notify_questionnaire_required(
                        vacancy.title, vacancy.url, callback.from_user.id
                    )
                    if callback.message:
                        await callback.message.edit_reply_markup(reply_markup=None)
                else:
                    await self.notify(f"✗ {result.message}", callback.from_user.id)
        else:
            result = self.approval_service.skip(job_id, callback.from_user.id)
            await self._answer_callback(
                callback, result.message, show_alert=not result.ok
            )
            if result.ok and callback.message:
                await callback.message.edit_reply_markup(reply_markup=None)

    @staticmethod
    async def _answer_callback(
        callback: CallbackQuery, text: str, *, show_alert: bool = False
    ) -> None:
        try:
            await callback.answer(text, show_alert=show_alert)
        except TelegramAPIError:
            logger.warning("telegram_callback_answer_failed")

    async def _text_handler(self, message: Message) -> None:
        if not self.authorized(message.from_user.id):
            await message.answer(PRIVATE_REPLY)
            return
        text = message.text
        if text is None or text.startswith("/"):
            return
        future = self._captcha_futures.get(message.from_user.id)
        if future is not None and not future.done():
            future.set_result(text.strip())
            await message.answer("CAPTCHA input received.")
            return

    def _rich_card(self, vacancy: Vacancy) -> InputRichMessage:
        def value(name: str, default: str) -> str:
            result = getattr(vacancy, name, "")
            return result.strip() if isinstance(result, str) and result.strip() else default

        blocks = [
            InputRichBlockParagraph(
                text=vacancy.title,
            ),
            InputRichBlockParagraph(
                text=RichTextUrl(
                    text=RichTextBold(
                        text="Открыть вакансию",
                    ),
                    url=vacancy.url,
                ),
            ),
            InputRichBlockDivider(),
            InputRichBlockSectionHeading(
                text="Компания",
                size=1,
            ),
            InputRichBlockDetails(
                summary="Компания",
                blocks=[
                    InputRichBlockParagraph(
                        text=f"Компания: {vacancy.company}",
                    ),
                ],
            ),
            InputRichBlockDetails(
                summary="Зарплата",
                blocks=[
                    InputRichBlockParagraph(
                        text=f"Зарплата: {value('salary', 'Не указана')}",
                    ),
                ],
            ),
            InputRichBlockDetails(
                summary="Опыт работы",
                blocks=[
                    InputRichBlockParagraph(
                        text=f"Опыт работы: {value('experience', 'Не указан')}",
                    ),
                ],
            ),
        ]

        return InputRichMessage(
            blocks=blocks,
        )

    async def send_preview(
        self, vacancy: Vacancy, include_actions: bool, *, telegram_user_id: int
    ) -> None:
        target_user_id = telegram_user_id
        keyboard = None
        if include_actions:
            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="Откликнуться",
                            callback_data=f"apply:{vacancy.id}",
                        ),
                        InlineKeyboardButton(
                            text="Пропустить",
                            callback_data=f"skip:{vacancy.id}",
                        ),
                    ],
                ]
            )

        try:
            await self.bot.send_rich_message(
                chat_id=target_user_id,
                rich_message=self._rich_card(vacancy),
                reply_markup=keyboard,
            )
        except TelegramBadRequest:
            logger.warning("telegram_rich_message_fallback job_id=%s", vacancy.id)
            await self.bot.send_message(
                chat_id=target_user_id,
                text=self._html_card(vacancy),
                parse_mode="HTML",
                reply_markup=keyboard,
                disable_web_page_preview=True,
            )

        logger.info(
            "%s job_id=%s user=%s",
            "approval_requested" if include_actions else "preview_sent",
            vacancy.id,
            target_user_id,
        )

    def _html_card(self, vacancy: Vacancy) -> str:
        lines = [
            f"<b>{html.escape(vacancy.title)}</b>",
            "",
            f'<a href="{html.escape(vacancy.url, quote=True)}">Открыть вакансию</a>',
            "",
            f"<b>Компания:</b> {html.escape(vacancy.company or 'Не указана')}",
            f"<b>Зарплата:</b> {html.escape(vacancy.salary or 'Не указана')}",
            f"<b>Опыт:</b> {html.escape(vacancy.experience or 'Не указан')}",
        ]
        return "\n".join(lines)

    async def notify(self, text: str, *, telegram_user_id: int) -> None:
        target_user_id = telegram_user_id
        try:
            await self.bot.send_message(chat_id=target_user_id, text=text)
        except Exception as exc:
            logger.warning("telegram_notify_failed error=%s", exc)

    async def notify_questionnaire_required(
        self, title: str, url: str, *, telegram_user_id: int
    ) -> None:
        target_user_id = telegram_user_id
        text = (
            "📋 <b>Требуется ручной отклик (тестовое / анкета)</b>\n"
            "<b>" + html.escape(title) + "</b>\n"
            + '<a href="' + html.escape(url, quote=True) + '">Открыть вакансию на HH.ru</a>\n\n'
            + "Работодатель требует заполнении анкеты или выполнение тестового задания."
        )
        try:
            await self.bot.send_message(
                chat_id=target_user_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as exc:
            logger.warning("notify_questionnaire_failed error=%s", exc)

    async def request_captcha(
        self, screenshot: Path, title: str, timeout_seconds: int, *, telegram_user_id: int
    ) -> str | None:
        target_user_id = telegram_user_id
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._captcha_futures[target_user_id] = future
        try:
            await self.bot.send_photo(
                chat_id=target_user_id,
                photo=FSInputFile(screenshot),
                caption="CAPTCHA detected for: " + title + "\nReply with the text or use /cancel.",
            )
            return await asyncio.wait_for(future, timeout_seconds)
        except TimeoutError:
            return None
        finally:
            # Stage 4\7: remove only this user's entry, and only if it is
            # still the same future — a newer request must not be dropped.
            if self._captcha_futures.get(target_user_id) is future:
                del self._captcha_futures[target_user_id]

    async def check_updates(self, notify: bool = True, telegram_user_id: int | None = None) -> ReleaseInfo | None:
        try:
            release = await check_github_release(current_version=__version__)
        except Exception as exc:
            logger.debug("update_check_failed error=%s", exc)
            return None
        if release is not None:
            self.latest_release = release
            if notify and self._notified_release_tag != release.tag_name:
                self._notified_release_tag = release.tag_name
                # Stage 4\3: notify only if explicit telegram_user_id provided
                # (per-user broadcast belongs to 4\4)
                if telegram_user_id is not None:
                    await self.notify_update_available(release, telegram_user_id)
        return release

    async def notify_update_available(self, release: ReleaseInfo, telegram_user_id: int) -> None:
        text = (
            "🚀 <b>Доступно обновление HH.kz Agent " + html.escape(release.tag_name) + "!</b>\n"
            "Текущая версия: <code>v" + __version__ + "</code>\n\n"
            + '<a href="' + html.escape(release.html_url, quote=True) + '">Посмотреть список изменений на GitHub</a>'
        )
        try:
            await self.bot.send_message(
                chat_id=telegram_user_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as exc:
            logger.warning("notify_update_available_failed error=%s", exc)

    async def start_polling(self) -> None:
        while True:
            try:
                await self.dispatcher.start_polling(self.bot)
                break
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("telegram_polling_error error=%s, retrying in 5s", exc)
                await asyncio.sleep(5)

    async def stop(self) -> None:
        session = getattr(self.bot, "session", None)
        if session is not None:
            await session.close()