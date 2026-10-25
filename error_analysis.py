"""Разбор ошибок победителя дня 4: TF-IDF и LinearSVC на том же test."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from load_data import VALID_LABELS
from preprocess import LABEL_TO_ID, prepare_dataset
from train_baseline import split_dataset
from train_improved import make_linear_svc

ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURE_PATH = FIGURES_DIR / "confusion_matrix.png"
REPORT_PATH = ROOT / "reports" / "day05_error_analysis.md"
EXPECTED_NEGATIVE_RECALL = 0.4928
EXAMPLES_PER_PAIR = 2
LABELS = [LABEL_TO_ID[name] for name in VALID_LABELS]


def fit_and_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Учит LinearSVC на train и добавляет предсказания к test."""
    pipeline = make_linear_svc()
    pipeline.fit(train["text_clean"], train["label"])
    predicted = test.copy()
    predicted["predicted"] = pipeline.predict(test["text_clean"])
    return predicted


def build_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Матрица 3×3: строки — истина, столбцы — предсказание."""
    matrix = confusion_matrix(frame["label"], frame["predicted"], labels=LABELS)
    return pd.DataFrame(matrix, index=list(VALID_LABELS), columns=list(VALID_LABELS))


def save_heatmap(matrix: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Матрица ошибок LinearSVC")
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Истина")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def error_pairs(matrix: pd.DataFrame) -> list[dict]:
    """Ошибочные пары по убыванию числа ошибок."""
    pairs = []
    for true_name in VALID_LABELS:
        for predicted_name in VALID_LABELS:
            if true_name == predicted_name:
                continue
            pairs.append(
                {
                    "true": true_name,
                    "predicted": predicted_name,
                    "count": int(matrix.loc[true_name, predicted_name]),
                }
            )
    pairs.sort(key=lambda pair: pair["count"], reverse=True)
    return pairs


def sample_errors(frame: pd.DataFrame, true_name: str, predicted_name: str) -> pd.DataFrame:
    true_id = LABEL_TO_ID[true_name]
    predicted_id = LABEL_TO_ID[predicted_name]
    rows = frame[(frame["label"] == true_id) & (frame["predicted"] == predicted_id)]
    return rows.head(EXAMPLES_PER_PAIR)


def _check_negative_recall(matrix: pd.DataFrame) -> float:
    support = int(matrix.loc["negative"].sum())
    hits = int(matrix.loc["negative", "negative"])
    recall = hits / support if support else 0.0
    if round(recall, 4) != EXPECTED_NEGATIVE_RECALL:
        raise RuntimeError(
            "Recall negative не совпал с днём 4: "
            f"{recall:.4f} против {EXPECTED_NEGATIVE_RECALL:.4f}"
        )
    return recall


def write_report(matrix: pd.DataFrame, frame: pd.DataFrame, path: Path) -> None:
    """Пишет матрицу, примеры ошибок и выводы по слабым местам."""
    recall = _check_negative_recall(matrix)
    total = int(matrix.to_numpy().sum())
    correct = int(sum(matrix.loc[name, name] for name in VALID_LABELS))
    errors = total - correct
    pairs = error_pairs(matrix)
    boundary = int(matrix.loc["positive", "neutral"] + matrix.loc["neutral", "positive"])
    lines = [
        "# День 5. Разбор ошибок",
        "",
        "Разбирается победитель дня 4: TF-IDF на n-граммах 1–2 и LinearSVC. "
        "Обучение только на train, разбиение то же, что в днях 3–4.",
        "",
        f"На test {total} строк, верных предсказаний {correct}, ошибок {errors}. "
        f"Recall класса `negative`: {recall:.4f}.",
        "",
        "## Матрица ошибок",
        "",
        "Строки — истинный класс, столбцы — предсказание. "
        f"График: `reports/figures/{FIGURE_PATH.name}`.",
        "",
        "| Истина \\ предсказание | negative | neutral | positive | Всего |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in VALID_LABELS:
        counts = [int(matrix.loc[name, column]) for column in VALID_LABELS]
        lines.append(
            f"| {name} | {counts[0]} | {counts[1]} | {counts[2]} | {sum(counts)} |"
        )
    lines.extend(
        [
            "",
            "## Куда уходят ошибки",
            "",
            "| Истина | Предсказание | Ошибок | Доля всех ошибок |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for pair in pairs:
        share = pair["count"] / errors if errors else 0.0
        lines.append(
            f"| {pair['true']} | {pair['predicted']} | {pair['count']} | {share:.1%} |"
        )
    lines.extend(
        [
            "",
            (
                f"Граница `positive` и `neutral` даёт {boundary} из {errors} ошибок. "
                f"`negative` уходит в `positive` {int(matrix.loc['negative', 'positive'])} раз "
                f"и в `neutral` {int(matrix.loc['negative', 'neutral'])} раз. "
                f"Обратных ошибок `neutral` → `negative` — "
                f"{int(matrix.loc['neutral', 'negative'])}."
            ),
            "",
            "## Примеры",
            "",
            "Для каждой ошибочной пары взяты два первых текста test. Показан исходный текст.",
            "",
        ]
    )
    for pair in pairs:
        lines.append(f"### {pair['true']} → {pair['predicted']} ({pair['count']})")
        lines.append("")
        examples = sample_errors(frame, pair["true"], pair["predicted"])
        if examples.empty:
            lines.append("Таких ошибок нет.")
            lines.append("")
            continue
        for text in examples["text"].astype(str):
            lines.append(_fence(text))
            lines.append("")
    lines.extend(
        [
            "## Выводы",
            "",
            (
                f"Основной объём ошибок — граница `neutral` и `positive`: {boundary} из {errors}. "
                "Позитивная новость без яркого оценочного слова уходит в `neutral`. "
                "В примерах это рост цены InterContinental на спецдивиденд и заявка Asahi на бренды SABMiller: "
                "в тексте есть climbs и bid, но нет прямого «хорошо» или «плохо». "
                "Нейтральная новость со смешанными числами уходит в `positive`: "
                "продажи упали на 14% в Финляндии и выросли на 27% за рубежом, "
                "а мешок слов цепляется за up, growth и lifted."
            ),
            "",
            (
                "Отрицание и короткие тикерные сообщения ломают `negative`. "
                "«Does not want to go down» и «won't stay down» размечены как `positive`, "
                "а модель ставит `negative` из-за слова down. "
                "«$TWTR ... slowly falling» при этом предсказано как `positive`: рядом стоит best. "
                "«$GPS ... Sales -4%» уходит в `neutral`: падение задано числом, а не словом loss."
            ),
            "",
            (
                "Дальше полезнее ловить отрицание, смешанную полярность и короткие тикерные сообщения, "
                "чем снова расширять n-граммы. Веса классов поднимают recall `negative`, "
                "но день 4 уже показал цену: падает accuracy. Новый эксперимент в этот день не запускается."
            ),
            "",
            "Модель в файл не сохраняется: это шаг дня 6.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _fence(text: str) -> str:
    body = " ".join(text.split()).replace("```", "'''")
    return f"```\n{body}\n```"


def main() -> None:
    prepared, _stats = prepare_dataset()
    train, test = split_dataset(prepared)
    predicted = fit_and_predict(train, test)
    matrix = build_matrix(predicted)
    save_heatmap(matrix, FIGURE_PATH)
    write_report(matrix, predicted, REPORT_PATH)
    errors = int((predicted["label"] != predicted["predicted"]).sum())
    print(matrix.to_string())
    print(f"Ошибок: {errors} из {len(predicted)}")
    print(f"График: {FIGURE_PATH}")
    print(f"Отчёт: {REPORT_PATH}")


if __name__ == "__main__":
    main()
