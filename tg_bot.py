from __future__ import annotations

import asyncio
import html
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
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
from updater import ReleaseInfo, check_github_release
from version import __version__


logger = logging.getLogger(__name__)
PRIVATE_REPLY = "This bot is private."


@dataclass
class AgentControl:
    paused: bool = False
    circuit_reason: str = ""
    consecutive_search_errors: int = 0
    next_run_at: datetime | None = None
    wake_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


class TelegramService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        approval_service: ApprovalService,
        control: AgentControl,
        *,
        bot: Any | None = None,
        dispatcher: Dispatcher | None = None,
        now_factory: Callable[[], datetime] | None = None,
    ):
        self.settings = settings
        self.database = database
        self.approval_service = approval_service
        self.control = control
        self.bot = bot or Bot(token=settings.tg_bot_token)
        self.dispatcher = dispatcher or Dispatcher()
        self.now_factory = now_factory or (lambda: datetime.now().astimezone())
        self._captcha_future: asyncio.Future[str | None] | None = None
        self.latest_release: ReleaseInfo | None = None
        self._notified_release_tag: str | None = None
        self._register_handlers()

    def authorized(self, user_id: int) -> bool:
        user = self.database.get_user(user_id)
        return user is not None and user["is_active"] == 1

    def command(self, name: str, user_id: int) -> str:
        if not self.authorized(user_id):
            return PRIVATE_REPLY
        if name == "start":
            return "HH.kz Agent is ready. Use /status to inspect it."
        if name == "pause":
            self.control.paused = True
            self.control.next_run_at = None
            logger.info("agent_paused")
            return "Agent paused."
        if name == "resume":
            self.control.paused = False
            self.control.circuit_reason = ""
            self.control.consecutive_search_errors = 0
            self.control.next_run_at = None
            self.control.wake_event.set()
            logger.info("agent_resumed")
            return "Agent resumed."
        if name == "status":
            stats = self.database.stats(telegram_user_id=user_id)
            processed = sum(stats.values())
            version_str = f"v{__version__}"
            if self.latest_release is not None:
                version_str += f" (доступно обновление {self.latest_release.tag_name})"
            return (
                f"версия: {version_str}\n"
                f"mode: {self.settings.app_mode}\n"
                f"state: {'paused' if self.control.paused else 'running'}\n"
                f"processed: {processed}\n"
                f"applied today: {self.database.applied_today(self.now_factory(), telegram_user_id=user_id)}"
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
            if self._captcha_future and not self._captcha_future.done():
                self._captcha_future.set_result(None)
            return "Current input request cancelled."
        return "Unknown command."

    def _diagnostics(self, user_id: int) -> str:
        run = self.database.latest_search_run(telegram_user_id=user_id)
        version_str = f"v{__version__}"
        if self.latest_release is not None:
            version_str += f" (доступно: {self.latest_release.tag_name})"
        if run is None:
            return f"версия: {version_str}\nNo search diagnostics recorded."
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
            f"open ({self.control.circuit_reason})"
            if self.control.circuit_reason
            else "closed"
        )
        if self.control.paused:
            next_run = "paused"
        elif self.control.next_run_at is None:
            next_run = "после текущего цикла"
        else:
            next_run = self.control.next_run_at.astimezone().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        return (
            f"версия: {version_str}\n"
            f"последний цикл: {started.astimezone():%Y-%m-%d %H:%M:%S}\n"
            f"состояние: {run.state}\n"
            f"длительность: {int((finished - started).total_seconds())} с\n"
            f"запросы: {run.query_count}\n"
            f"найдено: {run.found_results}\n"
            f"новые: {run.new_vacancies}\n"
            f"дубли: {run.duplicates}\n"
            f"отклонено — фильтр: {run.rejected_by_filter}\n"
            f"причины: {reasons}\n"
            f"карточки: {run.telegram_cards}\n"
            f"ошибки: {run.error_count} ({errors})\n"
            f"последняя ошибка: {run.last_safe_error or 'нет'}\n"
            f"breaker: {breaker}\n"
            f"last breaker: {run.circuit_reason or 'none'}\n"
            f"следующий запуск: {next_run}"
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
        if not self.authorized(message.from_user.id):
            await message.answer(PRIVATE_REPLY)
            return
        await message.answer(self.command(name, message.from_user.id))

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
        if self._captcha_future and not self._captcha_future.done():
            self._captcha_future.set_result(text.strip())
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
            f"📋 <b>Требуется ручной отклик (тестовое / анкета)</b>\n"
            f"<b>{html.escape(title)}</b>\n"
            f'<a href="{html.escape(url, quote=True)}">Открыть вакансию на HH.ru</a>\n\n'
            f"Работодатель требует заполнении анкеты или выполнение тестового задания."
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
        self._captcha_future = loop.create_future()
        await self.bot.send_photo(
            chat_id=target_user_id,
            photo=FSInputFile(screenshot),
            caption=f"CAPTCHA detected for: {title}\nReply with the text or use /cancel.",
        )
        try:
            return await asyncio.wait_for(self._captcha_future, timeout_seconds)
        except TimeoutError:
            return None
        finally:
            self._captcha_future = None

    async def check_updates(self, notify: bool = True) -> ReleaseInfo | None:
        try:
            release = await check_github_release(current_version=__version__)
        except Exception as exc:
            logger.debug("update_check_failed error=%s", exc)
            return None
        if release is not None:
            self.latest_release = release
            if notify and self._notified_release_tag != release.tag_name:
                self._notified_release_tag = release.tag_name
                await self.notify_update_available(release)
        return release

    async def notify_update_available(self, release: ReleaseInfo) -> None:
        text = (
            f"🚀 <b>Доступно обновление HH.kz Agent {html.escape(release.tag_name)}!</b>\n"
            f"Текущая версия: <code>v{__version__}</code>\n\n"
            f'<a href="{html.escape(release.html_url, quote=True)}">Посмотреть список изменений на GitHub</a>'
        )
        try:
            await self.bot.send_message(
                chat_id=self.settings.tg_user_id,
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