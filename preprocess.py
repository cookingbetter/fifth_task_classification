"""Очистка текстов финансовой тональности и подготовка таблицы для обучения."""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from load_data import VALID_LABELS, load_dataset

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "reports" / "day02_preprocessing.md"

LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {label_id: label for label, label_id in LABEL_TO_ID.items()}
PREPARED_COLUMNS = (
    "text",
    "text_clean",
    "sentiment",
    "label",
    "char_count",
    "word_count",
    "dollar_count",
)


class PreparedDataset(NamedTuple):
    frame: pd.DataFrame
    stats: dict


def clean_text(text: str) -> str:
    """Нижний регистр, без HTML и ссылок, со схлопнутыми пробелами.

    Знаки ``$`` у тикеров и остальная пунктуация сохраняются.
    """
    cleaned = str(text).lower()
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def prepare_dataset(frame: pd.DataFrame | None = None) -> PreparedDataset:
    """Возвращает таблицу для обучения и счётчики удалённых строк.

    Пустой текст, один и тот же очищенный текст с разными метками и точные
    повторы очищенного текста в результат не попадают.
    """
    source = load_dataset() if frame is None else frame.copy()
    missing = [name for name in ("text", "sentiment") if name not in source.columns]
    if missing:
        found = ", ".join(map(str, source.columns))
        raise ValueError(
            f"Нет обязательных колонок: {', '.join(missing)}. Найдены: {found}"
        )

    work = source.loc[:, ["text", "sentiment"]].copy()
    work["text_clean"] = work["text"].map(clean_text)
    class_before = _class_counts(work["sentiment"])

    empty_mask = work["text_clean"].eq("")
    class_empty = _class_counts(work.loc[empty_mask, "sentiment"])
    work = work.loc[~empty_mask].copy()

    label_nunique = work.groupby("text_clean", sort=False)["sentiment"].nunique()
    conflict_texts = label_nunique[label_nunique > 1].index
    conflict_mask = work["text_clean"].isin(conflict_texts)
    class_conflict = _class_counts(work.loc[conflict_mask, "sentiment"])
    conflict_example = _conflict_example(work.loc[conflict_mask])
    work = work.loc[~conflict_mask].copy()

    duplicate_mask = work["text_clean"].duplicated(keep="first")
    class_duplicate = _class_counts(work.loc[duplicate_mask, "sentiment"])
    work = work.loc[~duplicate_mask].copy()

    unknown = sorted(set(work["sentiment"].astype(str)) - set(LABEL_TO_ID))
    if unknown:
        expected = ", ".join(LABEL_TO_ID)
        raise ValueError(
            f"Неизвестные метки: {', '.join(unknown)}. Ожидаются: {expected}"
        )

    work["label"] = work["sentiment"].map(LABEL_TO_ID).astype(int)
    work["char_count"] = work["text_clean"].str.len().astype(int)
    work["word_count"] = work["text_clean"].str.split().str.len().astype(int)
    work["dollar_count"] = work["text_clean"].str.count(r"\$").astype(int)
    prepared = work.loc[:, list(PREPARED_COLUMNS)].reset_index(drop=True)

    stats = {
        "n_before": int(len(source)),
        "n_empty": int(empty_mask.sum()),
        "n_conflict_texts": int(len(conflict_texts)),
        "n_conflict_rows": int(conflict_mask.sum()),
        "n_duplicate_rows": int(duplicate_mask.sum()),
        "n_after": int(len(prepared)),
        "class_before": class_before,
        "class_after": _class_counts(prepared["sentiment"]),
        "class_empty": class_empty,
        "class_conflict": class_conflict,
        "class_duplicate": class_duplicate,
        "conflict_example": conflict_example,
    }
    removed = stats["n_empty"] + stats["n_conflict_rows"] + stats["n_duplicate_rows"]
    if stats["n_before"] - removed != stats["n_after"]:
        raise RuntimeError("Счётчики удалений не сходятся с размером таблицы")
    return PreparedDataset(prepared, stats)


def _class_counts(labels: pd.Series) -> dict[str, int]:
    counts = labels.value_counts()
    return {label: int(counts.get(label, 0)) for label in VALID_LABELS}


def _conflict_example(conflict_rows: pd.DataFrame) -> dict | None:
    if conflict_rows.empty:
        return None
    text_clean = str(conflict_rows["text_clean"].iloc[0])
    sample = conflict_rows.loc[conflict_rows["text_clean"] == text_clean, ["text", "sentiment"]]
    pairs = [
        {"text": str(row["text"]), "sentiment": str(row["sentiment"])}
        for _, row in sample.iterrows()
    ]
    return {"text_clean": text_clean, "pairs": pairs}


def _pick_kept(raw: pd.DataFrame, prepared_texts: set[str], mask: pd.Series) -> pd.Series:
    candidates = raw.loc[mask]
    for _, row in candidates.iterrows():
        if clean_text(row["text"]) in prepared_texts:
            return row
    raise RuntimeError("Не найден пример, который остаётся после очистки")


def _collect_examples(raw: pd.DataFrame, prepared: pd.DataFrame) -> dict[str, pd.Series]:
    text = raw["text"].astype(str)
    prepared_texts = set(prepared["text_clean"])
    has_ticker = text.str.contains(r"\$[A-Za-z]", regex=True)
    has_url = text.str.contains(r"https?://", regex=True)
    return {
        "news": _pick_kept(raw, prepared_texts, ~has_ticker & ~has_url),
        "ticker": _pick_kept(raw, prepared_texts, has_ticker),
        "url": _pick_kept(raw, prepared_texts, has_url),
    }


def _check_prepared(prepared: pd.DataFrame, stats: dict) -> None:
    if prepared["text_clean"].eq("").any():
        raise RuntimeError("В подготовленной таблице остался пустой текст")
    if prepared["text_clean"].duplicated().any():
        raise RuntimeError("В подготовленной таблице остались повторы текста")
    conflicts = prepared.groupby("text_clean")["label"].nunique()
    if (conflicts > 1).any():
        raise RuntimeError("В подготовленной таблице остались конфликты меток")
    expected = prepared["sentiment"].map(LABEL_TO_ID)
    if not prepared["label"].eq(expected).all():
        raise RuntimeError("Числовые метки не совпадают с картой классов")
    example = stats["conflict_example"]
    if example is not None and example["text_clean"] in set(prepared["text_clean"]):
        raise RuntimeError("Конфликтный текст остался в подготовленной таблице")


def format_stats(stats: dict) -> str:
    lines = [
        f"Было строк: {stats['n_before']}",
        f"Пустой текст: {stats['n_empty']}",
        (
            f"Конфликт меток: {stats['n_conflict_rows']} строк, "
            f"{stats['n_conflict_texts']} текстов"
        ),
        f"Повторы текста: {stats['n_duplicate_rows']}",
        f"Осталось строк: {stats['n_after']}",
        "Классы до -> после:",
    ]
    for label in VALID_LABELS:
        before = stats["class_before"][label]
        after = stats["class_after"][label]
        lines.append(f"  {label}: {before} -> {after}")
    return "\n".join(lines)


def write_report(
    prepared: pd.DataFrame,
    stats: dict,
    examples: dict[str, pd.Series],
    path: Path,
) -> None:
    """Пишет правила очистки, счётчики и примеры до и после."""
    lines = [
        "# День 2. Предобработка",
        "",
        "Тексты приводятся к одному виду функцией `clean_text`: нижний регистр, "
        "удаление HTML и ссылок `http://` / `https://`, схлопывание пробелов. "
        "Знаки `$` у тикеров и остальная пунктуация сохраняются.",
        "",
        "В обучающую таблицу не попадают:",
        "",
        "- пустой текст после очистки;",
        "- один и тот же очищенный текст с разными метками, такие строки удаляются все;",
        "- точные повторы очищенного текста, остаётся первая строка.",
        "",
        "Метки кодируются так: "
        + ", ".join(f"`{label}` = {LABEL_TO_ID[label]}" for label in VALID_LABELS)
        + ". Исходное имя класса остаётся в колонке `sentiment`.",
        "",
        "На очищенном тексте считаются `char_count`, `word_count` и `dollar_count`. "
        "Это ещё не вектор для модели.",
        "",
        "## Сколько строк снято",
        "",
        f"- Было строк: {stats['n_before']}.",
        f"- Пустой текст: {stats['n_empty']}.",
        (
            f"- Конфликт меток: {stats['n_conflict_rows']} строк "
            f"({stats['n_conflict_texts']} уникальных текстов)."
        ),
        f"- Повторы текста: {stats['n_duplicate_rows']}.",
        f"- Осталось строк: {stats['n_after']}.",
        "",
        "## Классы",
        "",
        "| Класс | Код | Было | Конфликт | Повтор | Осталось |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in VALID_LABELS:
        lines.append(
            "| {label} | {code} | {before} | {conflict} | {duplicate} | {after} |".format(
                label=label,
                code=LABEL_TO_ID[label],
                before=stats["class_before"][label],
                conflict=stats["class_conflict"][label],
                duplicate=stats["class_duplicate"][label],
                after=stats["class_after"][label],
            )
        )
    negative_before = stats["class_before"]["negative"]
    negative_after = stats["class_after"]["negative"]
    negative_conflict = stats["class_conflict"]["negative"]
    negative_duplicate = stats["class_duplicate"]["negative"]
    lines.extend(
        [
            "",
            (
                f"Класс `negative` сократился с {negative_before} до {negative_after}. "
                f"Из снятых строк этого класса {negative_conflict} удалены из-за конфликта меток "
                f"и {negative_duplicate} как повторы. Строки не возвращаются: у конфликта нет "
                "надёжной единственной метки."
            ),
            "",
            "## Примеры",
            "",
        ]
    )
    titles = {
        "news": "Новость",
        "ticker": "Тикер",
        "url": "Ссылка",
    }
    for key, title in titles.items():
        row = examples[key]
        lines.extend(
            [
                f"### {title}",
                "",
                f"Класс: `{row['sentiment']}`",
                "",
                "Было:",
                "",
                _fence(row["text"]),
                "",
                "Стало:",
                "",
                _fence(clean_text(row["text"])),
                "",
            ]
        )
    lines.extend(["### Конфликт меток", ""])
    example = stats["conflict_example"]
    if example is None:
        lines.append("Конфликтных текстов в этой выгрузке нет.")
        lines.append("")
    else:
        labels = ", ".join(f"`{pair['sentiment']}`" for pair in example["pairs"])
        raw_texts = list(dict.fromkeys(pair["text"] for pair in example["pairs"]))
        lines.append(f"Метки одних и тех же данных: {labels}.")
        lines.append("")
        for raw_text in raw_texts:
            lines.append(_fence(raw_text))
            lines.append("")
        if any(clean_text(raw_text) != raw_text for raw_text in raw_texts):
            lines.extend(["После очистки:", "", _fence(example["text_clean"]), ""])
        lines.append("В подготовленной таблице этого текста нет.")
        lines.append("")
    lines.extend(
        [
            "## Колонки результата",
            "",
            ", ".join(f"`{name}`" for name in prepared.columns),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _fence(text: str) -> str:
    body = " ".join(str(text).split()).replace("```", "'''")
    return f"```\n{body}\n```"


def print_examples(examples: dict[str, pd.Series], stats: dict) -> None:
    titles = {"news": "Новость", "ticker": "Тикер", "url": "Ссылка"}
    for key, title in titles.items():
        row = examples[key]
        print(f"\n=== {title} ({row['sentiment']}) ===")
        print("Было:", " ".join(str(row["text"]).split()))
        print("Стало:", clean_text(row["text"]))
    example = stats["conflict_example"]
    if example is None:
        return
    labels = ", ".join(pair["sentiment"] for pair in example["pairs"])
    print(f"\n=== Конфликт ({labels}) ===")
    print("Стало:", example["text_clean"])
    print("В подготовленной таблице этого текста нет.")


def main() -> None:
    raw = load_dataset()
    prepared, stats = prepare_dataset(raw)
    examples = _collect_examples(raw, prepared)
    _check_prepared(prepared, stats)
    print(format_stats(stats))
    print_examples(examples, stats)
    write_report(prepared, stats, examples, REPORT_PATH)
    print(f"\nОтчёт: {REPORT_PATH}")


if __name__ == "__main__":
    main()
