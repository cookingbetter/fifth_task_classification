"""Три улучшения baseline на том же train/test, что и день 3."""

from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import prepare_dataset
from train_baseline import (
    LOGREG_MAX_ITER,
    RANDOM_STATE,
    RESULT_PATH,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    evaluate,
    make_baseline,
    split_dataset,
)

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "reports" / "day04_comparison.md"
TRIGRAM_RANGE = (1, 3)


def make_balanced() -> Pipeline:
    """Тот же TF-IDF, логистическая регрессия с весами классов."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=TFIDF_MAX_FEATURES,
                    ngram_range=TFIDF_NGRAM_RANGE,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=LOGREG_MAX_ITER,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_trigrams() -> Pipeline:
    """TF-IDF с триграммами и логистическая регрессия без весов классов."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=TFIDF_MAX_FEATURES,
                    ngram_range=TRIGRAM_RANGE,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=LOGREG_MAX_ITER, random_state=RANDOM_STATE),
            ),
        ]
    )


def make_linear_svc() -> Pipeline:
    """Тот же TF-IDF и линейный SVM."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=TFIDF_MAX_FEATURES,
                    ngram_range=TFIDF_NGRAM_RANGE,
                ),
            ),
            ("clf", LinearSVC(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )


def _read_saved_baseline(path: Path) -> float:
    line = path.read_text(encoding="utf-8").strip()
    return float(line.split(":")[1].strip())


def _negative(metrics: dict) -> dict:
    for row in metrics["per_class"]:
        if row["label"] == "negative":
            return row
    raise KeyError("В отчёте нет класса negative")


def _run(name: str, pipeline: Pipeline, train, test) -> dict:
    pipeline.fit(train["text_clean"], train["label"])
    metrics = evaluate(pipeline, test)
    negative = _negative(metrics)
    return {
        "name": name,
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "negative_recall": negative["recall"],
        "negative_f1": negative["f1"],
    }


def _pick_best(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: row["macro_f1"])


def write_report(rows: list[dict], saved_baseline: float, path: Path) -> None:
    """Пишет сравнение с baseline и объяснение победителя."""
    baseline = rows[0]
    best = _pick_best(rows)
    lines = [
        "# День 4. Сравнение с baseline",
        "",
        "Все варианты обучены на одном и том же разбиении, что и день 3: "
        "`prepare_dataset()`, `split_dataset()`, test 20%, `random_state=42`. "
        "Векторизатор каждого варианта учится только на train. "
        "Лучший вариант выбирается по macro F1 на этом test.",
        "",
        (
            f"Сохранённая точка отсчёта: {saved_baseline:.4f}. "
            f"Baseline, посчитанный заново: {baseline['macro_f1']:.4f}."
        ),
        "",
        "| Вариант | Macro F1 | Разница | Recall negative | F1 negative | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        delta = row["macro_f1"] - baseline["macro_f1"]
        delta_text = "—" if row is baseline else f"{delta:+.4f}"
        lines.append(
            "| {name} | {macro_f1:.4f} | {delta} | {negative_recall:.4f} | "
            "{negative_f1:.4f} | {accuracy:.4f} |".format(delta=delta_text, **row)
        )
    lines.extend(["", "## Лучший вариант", ""])
    if best is baseline:
        lines.append(
            "Лучшим остаётся baseline: ни одно из трёх изменений не подняло macro F1 на этом test."
        )
    else:
        delta = best["macro_f1"] - baseline["macro_f1"]
        recall_delta = best["negative_recall"] - baseline["negative_recall"]
        lines.append(
            f"Лучший вариант — {best['name']}: macro F1 {best['macro_f1']:.4f} "
            f"({delta:+.4f} к baseline). "
            f"Recall класса `negative` изменился с {baseline['negative_recall']:.4f} "
            f"до {best['negative_recall']:.4f} ({recall_delta:+.4f}). "
            f"F1 этого класса: {baseline['negative_f1']:.4f} → {best['negative_f1']:.4f}. "
            f"Accuracy: {baseline['accuracy']:.4f} → {best['accuracy']:.4f}."
        )
        higher_recall = [
            row
            for row in rows
            if row is not best and row["negative_recall"] > best["negative_recall"]
        ]
        if higher_recall:
            other = max(higher_recall, key=lambda row: row["negative_recall"])
            lines.extend(
                [
                    "",
                    (
                        f"У варианта «{other['name']}» recall `negative` выше "
                        f"({other['negative_recall']:.4f}), но macro F1 ниже "
                        f"({other['macro_f1']:.4f}): accuracy падает до {other['accuracy']:.4f}. "
                        f"«{best['name']}» поднимает recall `negative` и сохраняет accuracy, "
                        "поэтому macro F1 выше."
                    ),
                ]
            )
    lines.extend(["", "## Что помогло и что нет", ""])
    for row in rows[1:]:
        delta = row["macro_f1"] - baseline["macro_f1"]
        recall_delta = row["negative_recall"] - baseline["negative_recall"]
        accuracy_delta = row["accuracy"] - baseline["accuracy"]
        if round(delta, 4) > 0:
            verdict = "помогло"
        elif round(delta, 4) < 0:
            verdict = "не помогло"
        else:
            verdict = "не изменило macro F1"
        detail = (
            f"- {row['name']} — {verdict}: macro F1 {delta:+.4f}, "
            f"recall `negative` {recall_delta:+.4f}, accuracy {accuracy_delta:+.4f}."
        )
        if round(recall_delta, 4) == 0:
            detail += " Слабый класс `negative` этот вариант не сдвинул."
        lines.append(detail)
    lines.extend(
        [
            "",
            "Модель в файл не сохраняется: это шаг дня 6.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    prepared, _stats = prepare_dataset()
    train, test = split_dataset(prepared)
    experiments = [
        ("Baseline TF-IDF 1–2 + LogReg", make_baseline()),
        ("TF-IDF 1–2 + LogReg, class_weight=balanced", make_balanced()),
        ("TF-IDF 1–3 + LogReg", make_trigrams()),
        ("TF-IDF 1–2 + LinearSVC", make_linear_svc()),
    ]
    rows = [_run(name, pipeline, train, test) for name, pipeline in experiments]
    saved_baseline = _read_saved_baseline(RESULT_PATH)
    if round(rows[0]["macro_f1"], 4) != round(saved_baseline, 4):
        raise RuntimeError(
            "Повторный baseline не совпал с reports/baseline_result.txt: "
            f"{rows[0]['macro_f1']:.4f} против {saved_baseline:.4f}"
        )
    best = _pick_best(rows)
    write_report(rows, saved_baseline, REPORT_PATH)
    for row in rows:
        delta = row["macro_f1"] - rows[0]["macro_f1"]
        print(
            f"{row['name']}: macro F1 {row['macro_f1']:.4f} ({delta:+.4f}), "
            f"recall negative {row['negative_recall']:.4f}, accuracy {row['accuracy']:.4f}"
        )
    print(f"Лучший: {best['name']}")
    print(f"Отчёт: {REPORT_PATH}")


if __name__ == "__main__":
    main()
