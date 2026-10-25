"""Baseline: TF-IDF и логистическая регрессия на очищенных текстах."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from load_data import VALID_LABELS
from preprocess import ID_TO_LABEL, LABEL_TO_ID, prepare_dataset

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "reports" / "day03_baseline.md"
RESULT_PATH = ROOT / "reports" / "baseline_result.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
LOGREG_MAX_ITER = 1000


def make_baseline() -> Pipeline:
    """TF-IDF и логистическая регрессия. Векторизатор учится только на train."""
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
                LogisticRegression(max_iter=LOGREG_MAX_ITER, random_state=RANDOM_STATE),
            ),
        ]
    )


def split_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Стратифицированное разбиение. День 4 импортирует эту же функцию."""
    train, test = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame["label"],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def evaluate(pipeline: Pipeline, test: pd.DataFrame) -> dict:
    """Считает метрики и предсказания на test."""
    predicted = pipeline.predict(test["text_clean"])
    labels = [LABEL_TO_ID[name] for name in VALID_LABELS]
    target_names = [ID_TO_LABEL[label] for label in labels]
    precision, recall, f1, support = precision_recall_fscore_support(
        test["label"],
        predicted,
        labels=labels,
        zero_division=0,
    )
    per_class = []
    for name, prec, rec, score, count in zip(target_names, precision, recall, f1, support):
        per_class.append(
            {
                "label": name,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(score),
                "support": int(count),
            }
        )
    return {
        "predicted": predicted,
        "macro_f1": float(f1_score(test["label"], predicted, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(test["label"], predicted)),
        "per_class": per_class,
        "report": classification_report(
            test["label"],
            predicted,
            labels=labels,
            target_names=target_names,
            digits=4,
            zero_division=0,
        ),
    }


def _class_counts(labels: pd.Series) -> dict[str, int]:
    counts = labels.value_counts()
    return {ID_TO_LABEL[int(label)]: int(counts.get(label, 0)) for label in range(len(VALID_LABELS))}


def _example_rows(test: pd.DataFrame, predicted: list | pd.Series) -> list[dict]:
    frame = test.copy()
    frame["predicted"] = predicted
    examples = []
    for name in VALID_LABELS:
        label_id = LABEL_TO_ID[name]
        row = frame.loc[frame["label"] == label_id].iloc[0]
        examples.append(
            {
                "text": str(row["text_clean"]),
                "true": ID_TO_LABEL[int(row["label"])],
                "predicted": ID_TO_LABEL[int(row["predicted"])],
            }
        )
    return examples


def format_result(macro_f1: float) -> str:
    return f"Baseline Macro F1: {macro_f1:.4f}\n"


def write_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    metrics: dict,
    examples: list[dict],
    path: Path,
) -> None:
    """Пишет схему baseline, метрики на test и примеры предсказаний."""
    train_counts = _class_counts(train["label"])
    test_counts = _class_counts(test["label"])
    lines = [
        "# День 3. Baseline",
        "",
        "Цепочка: `text_clean` → TF-IDF → логистическая регрессия. "
        f"TF-IDF: до {TFIDF_MAX_FEATURES} признаков, n-граммы {TFIDF_NGRAM_RANGE[0]}–{TFIDF_NGRAM_RANGE[1]}. "
        "Словарь векторизатора строится только по train. "
        f"Логистическая регрессия: `max_iter={LOGREG_MAX_ITER}`, без весов классов.",
        "",
        f"Разбиение: test {TEST_SIZE:.0%}, `random_state={RANDOM_STATE}`, стратификация по `label`.",
        "",
        "## Размеры",
        "",
        f"- Train: {len(train)} строк.",
        f"- Test: {len(test)} строк.",
        "",
        "| Класс | Train | Test |",
        "| --- | ---: | ---: |",
    ]
    for name in VALID_LABELS:
        lines.append(f"| {name} | {train_counts[name]} | {test_counts[name]} |")
    lines.extend(
        [
            "",
            "## Метрики на test",
            "",
            f"Точка отсчёта для следующих дней — macro F1: **{metrics['macro_f1']:.4f}**.",
            "",
            (
                f"Accuracy: {metrics['accuracy']:.4f}. "
                "На этой выборке она выше macro F1, потому что `neutral` занимает большую часть теста."
            ),
            "",
            "| Класс | Precision | Recall | F1 | Support |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics["per_class"]:
        lines.append(
            "| {label} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {support} |".format(**row)
        )
    lines.extend(
        [
            "",
            "```",
            metrics["report"].rstrip(),
            "```",
            "",
            "## Примеры предсказаний",
            "",
            "По одному первому примеру каждого класса из test.",
            "",
        ]
    )
    for example in examples:
        lines.extend(
            [
                f"Истина: `{example['true']}`. Предсказание: `{example['predicted']}`.",
                "",
                _fence(example["text"]),
                "",
            ]
        )
    lines.append("Модель в файл не сохраняется: это шаг дня 6.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _fence(text: str) -> str:
    body = " ".join(text.split()).replace("```", "'''")
    return f"```\n{body}\n```"


def main() -> None:
    prepared, _stats = prepare_dataset()
    train, test = split_dataset(prepared)
    pipeline = make_baseline()
    pipeline.fit(train["text_clean"], train["label"])
    metrics = evaluate(pipeline, test)
    examples = _example_rows(test, metrics["predicted"])
    if "vocabulary_" not in pipeline.named_steps["tfidf"].__dict__:
        raise RuntimeError("TF-IDF не обучен")
    report_text = format_result(metrics["macro_f1"])
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(report_text, encoding="utf-8")
    write_report(train, test, metrics, examples, REPORT_PATH)
    print(report_text, end="")
    print(metrics["report"])
    print(f"Train: {len(train)}, test: {len(test)}")
    for example in examples:
        print(f"{example['true']} -> {example['predicted']}: {example['text'][:80]}")
    print(f"Отчёт: {REPORT_PATH}")
    print(f"Точка отсчёта: {RESULT_PATH}")


if __name__ == "__main__":
    main()
