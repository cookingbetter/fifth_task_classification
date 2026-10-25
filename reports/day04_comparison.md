# День 4. Сравнение с baseline

Все варианты обучены на одном и том же разбиении, что и день 3: `prepare_dataset()`, `split_dataset()`, test 20%, `random_state=42`. Векторизатор каждого варианта учится только на train. Лучший вариант выбирается по macro F1 на этом test.

Сохранённая точка отсчёта: 0.6832. Baseline, посчитанный заново: 0.6832.

| Вариант | Macro F1 | Разница | Recall negative | F1 negative | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline TF-IDF 1–2 + LogReg | 0.6832 | — | 0.2899 | 0.4348 | 0.7992 |
| TF-IDF 1–2 + LogReg, class_weight=balanced | 0.7103 | +0.0270 | 0.6377 | 0.5500 | 0.7815 |
| TF-IDF 1–3 + LogReg | 0.6900 | +0.0067 | 0.2899 | 0.4396 | 0.8065 |
| TF-IDF 1–2 + LinearSVC | 0.7333 | +0.0501 | 0.4928 | 0.5763 | 0.8054 |

## Лучший вариант

Лучший вариант — TF-IDF 1–2 + LinearSVC: macro F1 0.7333 (+0.0501 к baseline). Recall класса `negative` изменился с 0.2899 до 0.4928 (+0.2029). F1 этого класса: 0.4348 → 0.5763. Accuracy: 0.7992 → 0.8054.

У варианта «TF-IDF 1–2 + LogReg, class_weight=balanced» recall `negative` выше (0.6377), но macro F1 ниже (0.7103): accuracy падает до 0.7815. «TF-IDF 1–2 + LinearSVC» поднимает recall `negative` и сохраняет accuracy, поэтому macro F1 выше.

## Что помогло и что нет

- TF-IDF 1–2 + LogReg, class_weight=balanced — помогло: macro F1 +0.0270, recall `negative` +0.3478, accuracy -0.0177.
- TF-IDF 1–3 + LogReg — помогло: macro F1 +0.0067, recall `negative` +0.0000, accuracy +0.0073. Слабый класс `negative` этот вариант не сдвинул.
- TF-IDF 1–2 + LinearSVC — помогло: macro F1 +0.0501, recall `negative` +0.2029, accuracy +0.0062.

Модель в файл не сохраняется: это шаг дня 6.
