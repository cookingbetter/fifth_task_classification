"""Сохранение победителя и предсказание тональности новых текстов."""

from __future__ import annotations

from pathlib import Path

import joblib

from preprocess import ID_TO_LABEL, clean_text, prepare_dataset
from train_improved import make_linear_svc

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
MODEL_PATH = MODELS_DIR / "model.joblib"
REPORT_PATH = ROOT / "reports" / "day06_inference.md"

DEMO_TEXTS = [
    "Apple shares jump 5% after a record earnings report",
    "The company reported a quarterly loss and announced mass layoffs",
    "The board will meet on Thursday to discuss the agenda",
    "The stock won't stay down",
    "$AAPL Sales -4%",
]


def save_model() -> int:
    """Учит TF-IDF и LinearSVC на всей подготовленной таблице и пишет оба файла."""
    prepared, _stats = prepare_dataset()
    _ensure_new_texts(prepared["text"].astype(str), prepared["text_clean"].astype(str))
    pipeline = make_linear_svc()
    pipeline.fit(prepared["text_clean"], prepared["label"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline.named_steps["tfidf"], VECTORIZER_PATH)
    joblib.dump(pipeline.named_steps["clf"], MODEL_PATH)
    return len(prepared)


def predict_sentiment(texts: str | list[str]) -> list[str]:
    """Классифицирует новые тексты загруженными файлами, без обучения.

    На вход одна строка или список строк. На выход список имён классов:
    ``negative``, ``neutral`` или ``positive``.
    """
    if isinstance(texts, str):
        texts = [texts]
    cleaned = [clean_text(text) for text in texts]
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    predicted = model.predict(vectorizer.transform(cleaned))
    return [ID_TO_LABEL[int(label)] for label in predicted]


def _ensure_new_texts(raw_texts, clean_texts) -> None:
    raw_set = set(raw_texts)
    clean_set = set(clean_texts)
    for text in DEMO_TEXTS:
        if text in raw_set or clean_text(text) in clean_set:
            raise RuntimeError(f"Демонстрационный текст уже есть в датасете: {text}")


def write_report(n_rows: int, rows: list[dict], path: Path) -> None:
    """Описывает формат входа и выхода и пять предсказаний."""
    lines = [
        "# День 6. Предсказание на новых текстах",
        "",
        "Сохранён победитель дня 4: TF-IDF на n-граммах 1–2 и LinearSVC. "
        f"Оба файла получены из одного обучения на всей подготовленной таблице, {n_rows} строк. "
        "Это тот же алгоритм, что в днях 4–5, но другие веса: test-метрики описывают модель, "
        "обученную на 80% данных.",
        "",
        f"- Векторизатор: `models/{VECTORIZER_PATH.name}`",
        f"- Модель: `models/{MODEL_PATH.name}`",
        "",
        "У LinearSVC нет вероятностей классов. Ответ — имя класса.",
        "",
        "## Формат",
        "",
        "Вход `predict_sentiment`: одна строка или список строк на английском.",
        "",
        "Перед векторизацией каждый текст проходит `clean_text`: нижний регистр, "
        "удаление HTML и ссылок, схлопывание пробелов. Векторизатор только преобразует "
        "текст (`transform`), словарь заново не строится.",
        "",
        "Выход: список имён `negative`, `neutral`, `positive` в том же порядке, что и вход. "
        "Одна строка на входе тоже возвращает список из одного элемента.",
        "",
        "## Примеры",
        "",
        "Фразы новые, в CSV их нет. Предсказания получены загрузкой файлов с диска.",
        "",
        "| Текст | После очистки | Предсказание |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['text']} | {row['cleaned']} | `{row['label']}` |"
        )
    lines.extend(
        [
            "",
            "Фразы «won't stay down» и «Sales -4%» взяты из тех же ситуаций, что и ошибки дня 5. "
            "В таблице стоит фактический класс этой сохранённой модели.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    n_rows = save_model()
    labels = predict_sentiment(DEMO_TEXTS)
    rows = [
        {"text": text, "cleaned": clean_text(text), "label": label}
        for text, label in zip(DEMO_TEXTS, labels)
    ]
    write_report(n_rows, rows, REPORT_PATH)
    print(f"Обучено на {n_rows} строках")
    print(f"Векторизатор: {VECTORIZER_PATH}")
    print(f"Модель: {MODEL_PATH}")
    for row in rows:
        print(f"{row['label']}: {row['text']}")
    print(f"Отчёт: {REPORT_PATH}")


if __name__ == "__main__":
    main()
