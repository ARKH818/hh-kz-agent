# Безопасность hh-kz-agent

Этот документ описывает модель угроз Phase 5.7, защищаемые данные, реализованные механизмы и остаточные риски. Он не заявляет полную TOCTOU-proofness: защита снижает число доступных атак, но не устраняет любую возможность подмены путей во всех операциях.

## Поддерживаемая версия Python

Минимальная поддерживаемая версия — **Python 3.11**. Это основано на явной проверке в `setup_wizard.py` (`Python 3.11+`). `SETUP_CHECKLIST.md` рекомендует Python 3.13. Метаданные зависимостей допускают более старые версии, но приложение их не поддерживает.

`RedactingFilter` возвращает redacted копию `LogRecord`. В Python 3.12+ это поддерживается нативно в `Logger.handle` и `Handler.handle`. Для Python 3.11 `logging_setup.py` устанавливает узкий compatibility shim с той же семантикой: фильтр заменяет запись, не изменяя исходный `LogRecord`. Минимальная версия Python из-за этого не повышалась.

## Защищаемые данные

- **HH credentials** — токены и секреты HH.kz, данные авторизации и значения, которые могут попасть в диагностику.
- **Authentication/session data** — cookies, session state браузерных профилей, состояние входа HH, идентификаторы сессий.
- **`profile.yaml`** — персональные данные кандидата, резюме, поисковые параметры и настройки откликов.
- **`.env`** — Telegram bot token, user id, ключи и локальные секреты.
- **SQLite database и side-files** — `agent.db`, `-journal`, `-wal`, `-shm`, `-sqlite`, `-tmp`; состояние откликов, подтверждений и привязок браузерных профилей.
- **Browser profiles** — persistent profile directories, включая cookies, localStorage, IndexedDB, CacheStorage и Service Worker state.
- **Application logs** — диагностика, из которой удаляются токены, пароли, Authorization/Cookie заголовки, Bearer values, secret-bearing URL параметры и `client_id`; tracebacks заменяются управляемыми идентификаторами ошибок.

## Модель угроз

Защита предполагает, что **у атакующего нет права записи в родительские каталоги чувствительных файлов и каталогов**. Подмена symlink/junction, создание файлов в чужом каталоге и перенаправление пути считаются основной filesystem-угрозой. Код проверяет компоненты пути до I/O и, где возможно, использует уже проверенный handle вместо повторного открытия по имени.

Если атакующий может писать в родительский каталог, он способен менять компоненты после проверки. Handle-based операции снижают риск для контролируемых операций, но не гарантируют атомарную защиту всех внутренних open-операций SQLite и браузера.

### Другой локальный процесс того же пользователя

Процесс с тем же UID/SID действует с теми же правами, что и приложение. Ограничения прав доступа (POSIX mode, Windows DACL) не изолируют его от файлов пользователя. Такой процесс может читать или изменять `.env`, `profile.yaml`, базу данных, логи и профиль браузера напрямую, а также влиять на окружение процесса. Phase 5.7 не решает эту задачу: нужны отдельные account isolation, trusted OS boundaries или шифрование с внешним ключом.

### Процесс с правами администратора

Администратор (root, SYSTEM/Administrator) обходит обычные file permissions и ACL, может читать память процесса, подменять библиотеки и перехватывать I/O. Данная модель угроз не защищает от администратора или компрометации ОС.

## Реализованные механизмы

### POSIX

- `validate_path_before_io()` проверяет “сырые” компоненты до lexical normalization.
- `ensure_validated_directory()` создаёт каталоги только через уже проверенный parent directory handle.
- Каждый каталог открывается через `dirfd` с `O_NOFOLLOW | O_DIRECTORY | O_CLOEXEC`; конечный файл открывается относительно проверенного parent fd.
- Тип и права проверяются через `fstat()`/`fchmod()` на открытом объекте, а не по имени path.

### Windows

- Операции выполняются через `NtCreateFile(RootDirectory=...)` с `FILE_OPEN_REPARSE_POINT`; каждый компонент проверяется по handle.
- `GetFileInformationByHandle()` отклоняет reparse point и несоответствие типа “directory/regular file”.
- ACL применяются через `SetSecurityInfo()` к уже открытому handle; для обратной совместимости сохранён path-based вызов `_apply_windows_acl`.
- DACL содержит только текущего пользователя, SYSTEM и Administrators; директории получают наследуемые ACE.

### Reparse point rejection

Symlink, junction и другие reparse pointы отклоняются на уровне path preflight и повторно на уровне открытого handle. Для SQLite side-files используется `os.scandir()` плюс `DirEntry.stat(follow_symlinks=False)`; подозрительный объект вызывает fail closed до protection.

### Validated file operations

- `open_validated_file()`, `read_validated_text()`, `write_validated_text()` и `ensure_validated_directory()` связывают проверку с конкретным объектом.
- `.env` и `profile.yaml` читаются и пишутся через проверенный handle, а не через `read_text()`/`write_text()` по path.
- Начальное открытие log file использует `_ValidatedRotatingFileHandler`.

### SQLite protection

- Разрешённые URI классифицируются; file-backed target проверяется до и после создания каталогов.
- Directory, database и существующие side-files защищаются через handle-based `protect_file`/`protect_directory`.
- На время `sqlite3.connect()` держится проверенный guard handle; после connect сравниваются `(st_dev, st_ino)` path и guard.

### Log redaction

- Один `RedactingFilter` подключается и к root logger, и к каждому handler.
- Фильтр возвращает копию `LogRecord`, не изменяя исходный `msg`, `args`, `exc_info` и `stack_info`.
- На Python 3.11 включён shim, повторяющий замену `LogRecord` из 3.12.
- `client_id` redacted всегда, без проверки context window.

### Browser profile protection

- Root каталог профилей проверяется до создания; каждый account profile создаётся через validated directory.
- Существующий profile защищается рекурсивно, но не перезаписывается.
- Cleanup logout работает только с HH origin и не удаляет persistent profile files.

## Остаточные риски

1. **SQLite path-based connect и WAL/SHM/journal.** Stdlib `sqlite3.connect()` открывает файл по имени и сам создаёт `-journal`, `-wal`, `-shm` и другие side-files. Guard handle защищает проверенный database object и обнаруживает его подмену, но не связывает все внутренние open-операции SQLite с handle. Атакующий с правом записи в родительский каталог может попытаться перехватить создание side-file.

2. **Browser internal file operations.** Playwright/CloakBrowser сами открывают файлы внутри profile directory после того, как каталог был проверен. Нет гарантии, что каждый внутренний open использует проверенный handle; защита ограничена каталогной проверкой и filesystem ACL.

3. **RotatingFileHandler rollover.** Начальное открытие лога handle-validated. Однако `doRollover()` stdlib работает с путями (`rename`, `open` backup-файлов). При записи в родительский каталог возможна гонка с этими path-based операциями.

4. **Ограничения платформенных тестов.** Текущая Windows-среда политикой запрещает создание symlink, поэтому соответствующие тесты пропускаются как `unverified/skipped`. Windows junction-тесты выполняются и покрывают reparse point. POSIX-тесты на Windows не выполняются и не должны считаться пройденными. Python 3.11 runtime здесь недоступен: совместимость подтверждена AST-проверкой под 3.11 и тестами compatibility-функций, но не полным запуском suite под 3.11.

Этот документ описывает текущее состояние Phase 5.7 и не является заверением об отсутствии уязвимостей.
