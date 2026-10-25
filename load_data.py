"""Загрузка и проверка датасета финансовой тональности."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_PATH = ROOT / "data" / "data.csv"

REQUIRED_COLUMNS = ("text", "sentiment")
VALID_LABELS = ("negative", "neutral", "positive")
SOURCE_COLUMNS = {"Sentence": "text", "Sentiment": "sentiment"}


def load_dataset(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Читает CSV и приводит колонки к схеме text / sentiment."""
    path = Path(path)
    frame = pd.read_csv(path)
    frame = frame.rename(columns=SOURCE_COLUMNS)
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        found = ", ".join(map(str, frame.columns))
        raise ValueError(
            f"В {path} нет обязательных колонок: {', '.join(missing)}. Найдены: {found}"
        )
    labels = set(frame["sentiment"].dropna().astype(str).unique())
    unknown = sorted(labels - set(VALID_LABELS))
    if unknown:
        expected = ", ".join(VALID_LABELS)
        raise ValueError(
            f"Неизвестные метки: {', '.join(unknown)}. Ожидаются: {expected}"
        )
    return frame.loc[:, list(REQUIRED_COLUMNS)].copy()


def summarize(frame: pd.DataFrame) -> dict:
    """Сводка по размеру, пропускам, классам, длинам и конфликтам меток."""
    text = frame["text"].astype("string")
    char_len = text.str.len()
    word_count = text.str.split().str.len()
    class_counts = frame["sentiment"].value_counts(dropna=False)
    n_rows = len(frame)
    by_class = (
        frame.assign(char_len=char_len, word_count=word_count)
        .groupby("sentiment", dropna=False)[["char_len", "word_count"]]
        .median()
    )
    label_nunique = frame.groupby(text, dropna=False)["sentiment"].nunique()

    return {
        "n_rows": int(n_rows),
        "n_columns": int(frame.shape[1]),
        "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "nulls": {name: int(count) for name, count in frame.isna().sum().items()},
        "class_counts": {str(label): int(count) for label, count in class_counts.items()},
        "class_shares": {
            str(label): (float(count) / n_rows if n_rows else 0.0)
            for label, count in class_counts.items()
        },
        "char_len": _series_stats(char_len),
        "word_count": _series_stats(word_count),
        "median_by_class": {
            str(label): {
                "char_len": float(row["char_len"]),
                "word_count": float(row["word_count"]),
            }
            for label, row in by_class.iterrows()
        },
        "n_unique_texts": int(text.nunique(dropna=False)),
        "n_duplicate_rows": int(frame["text"].duplicated().sum()),
        "n_conflicting_texts": int((label_nunique > 1).sum()),
        "ticker_share": float(text.str.contains(r"\$", regex=True, na=False).mean()),
        "url_share": float(text.str.contains(r"https?://", regex=True, na=False).mean()),
        "empty_texts": int(text.fillna("").str.strip().eq("").sum()),
    }


def _series_stats(series: pd.Series) -> dict[str, float]:
    described = series.astype(float).describe()
    return {
        "count": float(described["count"]),
        "mean": float(described["mean"]),
        "std": float(described["std"]),
        "min": float(described["min"]),
        "p25": float(described["25%"]),
        "median": float(described["50%"]),
        "p75": float(described["75%"]),
        "max": float(described["max"]),
    }


def format_summary(summary: dict) -> str:
    """Текстовая сводка для печати в консоль."""
    lines = [
        f"Строк: {summary['n_rows']}, колонок: {summary['n_columns']}",
        "Типы: " + ", ".join(f"{name}={dtype}" for name, dtype in summary["dtypes"].items()),
        "Пропуски: " + ", ".join(f"{name}={count}" for name, count in summary["nulls"].items()),
        "Классы:",
    ]
    for label in VALID_LABELS:
        count = summary["class_counts"].get(label, 0)
        share = summary["class_shares"].get(label, 0.0)
        lines.append(f"  {label}: {count} ({share:.1%})")
    char_len = summary["char_len"]
    words = summary["word_count"]
    lines.extend(
        [
            (
                "Длина в символах: "
                f"медиана {char_len['median']:.0f}, среднее {char_len['mean']:.1f}, "
                f"min {char_len['min']:.0f}, max {char_len['max']:.0f}"
            ),
            (
                "Длина в словах: "
                f"медиана {words['median']:.0f}, среднее {words['mean']:.1f}, "
                f"min {words['min']:.0f}, max {words['max']:.0f}"
            ),
            f"Уникальных текстов: {summary['n_unique_texts']}",
            f"Повторяющихся строк текста: {summary['n_duplicate_rows']}",
            f"Текстов с конфликтом меток: {summary['n_conflicting_texts']}",
            f"Пустых текстов: {summary['empty_texts']}",
            f"Доля текстов с $: {summary['ticker_share']:.1%}",
            f"Доля текстов со ссылкой: {summary['url_share']:.1%}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    frame = load_dataset()
    print(format_summary(summarize(frame)))


if __name__ == "__main__":
    main()
