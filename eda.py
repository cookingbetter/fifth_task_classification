"""Первичный анализ датасета финансовой тональности."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from load_data import VALID_LABELS, format_summary, load_dataset, summarize

ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "reports" / "figures"
REPORT_PATH = ROOT / "reports" / "day01_observations.md"
EXAMPLES_PER_CLASS = 2


def sample_examples(frame: pd.DataFrame, per_class: int = EXAMPLES_PER_CLASS) -> dict[str, list[str]]:
    """Берёт примеры каждого класса так, чтобы по возможности показать оба стиля текста."""
    examples: dict[str, list[str]] = {}
    for label in VALID_LABELS:
        subset = frame.loc[frame["sentiment"] == label, "text"].astype(str)
        if subset.empty:
            examples[label] = []
            continue
        chosen = [subset.iloc[0]]
        if per_class > 1 and len(subset) > 1:
            has_ticker = subset.str.contains(r"\$", regex=True)
            pool = subset[~has_ticker] if "$" in chosen[0] else subset[has_ticker]
            pool = pool[pool != chosen[0]]
            second = pool.iloc[0] if not pool.empty else subset.iloc[1]
            chosen.append(second)
        examples[label] = chosen
    return examples


def conflicting_example(frame: pd.DataFrame) -> tuple[str, list[str]] | None:
    """Возвращает один текст, которому в данных присвоены разные метки."""
    label_nunique = frame.groupby(frame["text"].astype(str), dropna=False)["sentiment"].nunique()
    conflicts = label_nunique[label_nunique > 1]
    if conflicts.empty:
        return None
    text = str(conflicts.index[0])
    labels = frame.loc[frame["text"].astype(str) == text, "sentiment"].astype(str).unique().tolist()
    return text, labels


def save_figures(frame: pd.DataFrame, figures_dir: Path) -> list[Path]:
    """Сохраняет распределение классов и длины текстов."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    work = frame.copy()
    work["char_len"] = work["text"].astype(str).str.len()
    work["word_count"] = work["text"].astype(str).str.split().str.len()
    order = list(VALID_LABELS)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=work, x="sentiment", order=order, ax=ax, color="#4C72B0")
    ax.set_title("Распределение классов")
    ax.set_xlabel("Тональность")
    ax.set_ylabel("Количество")
    paths.append(_save(fig, figures_dir / "sentiment_distribution.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(work["char_len"], bins=50, color="#4C72B0", edgecolor="white")
    ax.set_title("Распределение длины текста")
    ax.set_xlabel("Длина текста, символы")
    ax.set_ylabel("Количество")
    paths.append(_save(fig, figures_dir / "text_length_distribution.png"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(data=work, x="sentiment", y="char_len", order=order, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Длина в символах")
    axes[0].set_xlabel("Тональность")
    axes[0].set_ylabel("Символы")
    sns.boxplot(data=work, x="sentiment", y="word_count", order=order, ax=axes[1], color="#55A868")
    axes[1].set_title("Длина в словах")
    axes[1].set_xlabel("Тональность")
    axes[1].set_ylabel("Слова")
    fig.suptitle("Длины текстов по классам")
    paths.append(_save(fig, figures_dir / "length_by_class.png"))
    return paths


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_report(
    summary: dict,
    examples: dict[str, list[str]],
    conflict: tuple[str, list[str]] | None,
    figure_paths: list[Path],
    path: Path,
) -> None:
    """Пишет отчёт: числа прогона и наблюдения о структуре данных."""
    char_len = summary["char_len"]
    words = summary["word_count"]
    lines = [
        "# День 1. Наблюдения о данных",
        "",
        "Задача — классификация тональности финансового текста на три класса: "
        "`negative`, `neutral`, `positive`. Строки на этом этапе не удаляются и не очищаются.",
        "",
        "## Размер и схема",
        "",
        f"- Строк: {summary['n_rows']}, колонок после загрузки: {summary['n_columns']} (`text`, `sentiment`).",
        "- Исходные имена колонок: `Sentence`, `Sentiment`.",
        "- Типы: " + ", ".join(f"`{name}` — {dtype}" for name, dtype in summary["dtypes"].items()) + ".",
        "- Пропуски: " + ", ".join(f"`{name}` — {count}" for name, count in summary["nulls"].items()) + ".",
        f"- Пустых текстов: {summary['empty_texts']}.",
        "",
        "## Классы",
        "",
        "| Класс | Количество | Доля | Медиана символов | Медиана слов |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in VALID_LABELS:
        count = summary["class_counts"].get(label, 0)
        share = summary["class_shares"].get(label, 0.0)
        medians = summary["median_by_class"].get(label, {})
        lines.append(
            f"| {label} | {count} | {share:.1%} | {medians.get('char_len', 0):.0f} | {medians.get('word_count', 0):.0f} |"
        )
    lines.extend(
        [
            "",
            "## Длины текстов",
            "",
            (
                f"- Символы: медиана {char_len['median']:.0f}, среднее {char_len['mean']:.1f}, "
                f"минимум {char_len['min']:.0f}, максимум {char_len['max']:.0f}."
            ),
            (
                f"- Слова: медиана {words['median']:.0f}, среднее {words['mean']:.1f}, "
                f"минимум {words['min']:.0f}, максимум {words['max']:.0f}."
            ),
            "- Длины по классам близки: отдельный класс не выделяется длиной текста.",
            "",
            "## Качество разметки и состав текстов",
            "",
            f"- Уникальных текстов: {summary['n_unique_texts']}.",
            f"- Дополнительных строк с уже встреченным текстом: {summary['n_duplicate_rows']}.",
            f"- Текстов, у которых в данных больше одной метки: {summary['n_conflicting_texts']}.",
            f"- Доля текстов с символом `$`: {summary['ticker_share']:.1%}.",
            f"- Доля текстов со ссылкой `http`: {summary['url_share']:.1%}.",
            "",
        ]
    )
    if conflict is not None:
        text, labels = conflict
        quoted = "`, `".join(labels)
        lines.extend(
            [
                "Пример конфликта меток "
                f"(`{quoted}`):",
                "",
                f"> {_quote(text)}",
                "",
            ]
        )
    lines.extend(["## Примеры", ""])
    for label in VALID_LABELS:
        lines.append(f"### {label}")
        lines.append("")
        for text in examples.get(label, []):
            lines.append(f"> {_quote(text)}")
            lines.append("")
    lines.extend(
        [
            "## Графики",
            "",
            *[f"- `{figure.relative_to(ROOT)}`" for figure in figure_paths],
            "",
            "## Выводы",
            "",
            "Предсказывать нужно тональность короткого финансового текста. Три класса заданы явно, пропусков нет.",
            "",
            "Классы несбалансированы: `neutral` — чуть больше половины выборки, `negative` — меньшинство, около 15%. "
            "Accuracy будет завышена за счёт нейтрального класса, поэтому на следующих этапах ориентир — macro F1.",
            "",
            "Разметка шумная: сотни одинаковых текстов имеют сразу две метки, и во всех таких случаях это пара "
            "`negative` и `neutral`. Других конфликтов классов нет. Повторы на дне 1 оставлены: "
            "это наблюдение для подготовки данных, а не правка выборки.",
            "",
            "В одной таблице смешаны два стиля. Есть развёрнутые новости о компаниях и короткие сообщения с тикерами "
            "и ссылками. Очистка должна сохранять оба типа, а не рассчитывать только на связный новостной абзац.",
            "",
            "Тексты короткие: медиана около ста символов и меньше двадцати слов, длиннее 315 символов строк нет. "
            "Для baseline достаточно мешка слов или TF-IDF, длинный контекст здесь не требуется.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _quote(text: str) -> str:
    return " ".join(text.split())


def print_examples(examples: dict[str, list[str]]) -> None:
    for label in VALID_LABELS:
        print(f"\n=== {label} ===")
        for index, text in enumerate(examples.get(label, []), start=1):
            print(f"{index}. {_quote(text)}")


def main() -> None:
    frame = load_dataset()
    summary = summarize(frame)
    examples = sample_examples(frame)
    print(format_summary(summary))
    print_examples(examples)
    figure_paths = save_figures(frame, FIGURES_DIR)
    write_report(summary, examples, conflicting_example(frame), figure_paths, REPORT_PATH)
    print(f"\nОтчёт: {REPORT_PATH}")
    for figure in figure_paths:
        print(f"График: {figure}")


if __name__ == "__main__":
    main()
