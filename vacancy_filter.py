DEFAULT_STOP_WORDS = (
    "senior",
    "сеньор",
    "lead",
    "архитектор",
    "руководитель",
    "стажер",
    "intern",
    "trainee",
    "дизайнер",
    "маркетолог",
    "риелтор",
    "project manager",
    "project-manager",
    "general manager",
    "генеральный директор",
    "исполнительный директор",
    "executive director",
    "комплектовщик",
    "упаковщик",
    "документооборот",
    "разработчик",
    "программист",
    "data analyst",
    "аналитик данных",
    "водитель",
    "кладовщик",
    "заведующий складом",
    "начальник склада",
)

LOGISTICS_TITLE_KEYWORDS = (
    "логист",
    "логистика",
    "перевоз",
    "транспорт",
    "экспедитор",
    "экспедиц",
    "диспетчер",
    "вэд",
    "тамож",
    "supply chain",
    "supply-chain",
    "freight",
)

COMMERCIAL_TITLE_KEYWORDS = (
    "продаж",
    "sales",
    "sales manager",
    "account manager",
    "business development",
    "business developer",
    "bdm",
    "клиентск",
    "торгов",
)

REJECT_TITLE_KEYWORDS = (
    "генеральный директор",
    "исполнительный директор",
    "general manager",
    "executive director",
    "project manager",
    "project-manager",
    "production manager",
    "sales manager",
    "marketing manager",
    "product manager",
    "product owner",
    "разработчик",
    "программист",
    "аналитик данных",
    "data analyst",
    "дизайнер",
    "маркетолог",
    "риелтор",
    "водитель",
    "комплектовщик",
    "упаковщик",
    "кладовщик",
    "заведующий складом",
    "начальник склада",
    "документооборот",
)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().replace("ё", "е").split())


def title_rejection_reason(
    title: str,
    excluded_positions: tuple[str, ...],
) -> str | None:
    normalized = _normalize(title)

    for word in (*DEFAULT_STOP_WORDS, *REJECT_TITLE_KEYWORDS):
        if _normalize(word) in normalized:
            return word

    for word in excluded_positions:
        if _normalize(word) in normalized:
            return word

    return None


def is_relevant_logistics_title(title: str) -> bool:
    normalized = _normalize(title)

    if title_rejection_reason(title, ()) is not None:
        return False

    has_logistics_keyword = any(
        _normalize(keyword) in normalized
        for keyword in LOGISTICS_TITLE_KEYWORDS
    )

    if not has_logistics_keyword:
        return False

    has_commercial_keyword = any(
        _normalize(keyword) in normalized
        for keyword in COMMERCIAL_TITLE_KEYWORDS
    )

    if has_commercial_keyword:
        return False

    return True