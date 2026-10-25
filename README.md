# Финансовые новости: классификация тональности

Мини-проект проходит цикл text classification от CSV до предсказания на новом тексте. По английской финансовой новости или короткому рыночному сообщению модель ставит один из трёх классов: `negative`, `neutral`, `positive`.

Это не инвестиционная рекомендация.

## Данные

Файл `data/data.csv`, 5842 строки. В исходном CSV колонки называются `Sentence` и `Sentiment`. Загрузчик приводит их к `text` и `sentiment`.

`prepare_dataset()` очищает текст и оставляет 4803 строки. Снимаются пустые тексты, один и тот же текст с разными метками и точные повторы. Из-за конфликтов класс `negative` сокращается с 860 до 343 строк. Метки кодируются так: `negative` = 0, `neutral` = 1, `positive` = 2.

Очистка общая для обучения и inference: нижний регистр, удаление HTML и ссылок `http://` / `https://`, схлопывание пробелов. Знак `$` у тикеров сохраняется.

## Структура

- `load_data.py` — чтение CSV и схема `text` / `sentiment`
- `eda.py` — распределение классов и длины текстов
- `preprocess.py` — очистка, фильтры, числовые метки
- `train_baseline.py` — TF-IDF и логистическая регрессия
- `train_improved.py` — три улучшения и выбор лучшего варианта
- `error_analysis.py` — матрица ошибок и примеры
- `predict.py` — сохранение модели и предсказание новых текстов
- `data/data.csv` — исходный датасет
- `models/` — `vectorizer.joblib` и `model.joblib`
- `reports/` — отчёты и графики по шагам

## Установка

```bash
pip install -r requirements.txt
```

## Обучение

Команды идут по порядку. Все модели сравниваются на одном test: 20% строк, `random_state=42`, стратификация по классу. Векторизатор каждого прогона учится только на train.

```bash
python eda.py
python preprocess.py
python train_baseline.py
python train_improved.py
python error_analysis.py
```

- `eda.py` пишет `reports/day01_observations.md` и графики в `reports/figures/`.
- `preprocess.py` пишет `reports/day02_preprocessing.md`. Отдельный CSV не сохраняется.
- `train_baseline.py` пишет `reports/day03_baseline.md` и `reports/baseline_result.txt`.
- `train_improved.py` пишет `reports/day04_comparison.md`.
- `error_analysis.py` пишет `reports/day05_error_analysis.md` и `reports/figures/confusion_matrix.png`.

`python load_data.py` печатает сводку по сырому CSV без графиков.

## Inference

```bash
python predict.py
```

Команда учит TF-IDF и LinearSVC на всех 4803 подготовленных строках, записывает `models/vectorizer.joblib` и `models/model.joblib`, затем загружает эти файлы с диска и классифицирует пять демо-фраз. Пока скрипт идёт, словарь строится заново. Функция `predict_sentiment` после этого только читает файлы и вызывает `transform`.

```python
from predict import predict_sentiment

predict_sentiment("Apple shares jump 5% after a record earnings report")
predict_sentiment(["The board will meet on Thursday to discuss the agenda"])
```

Вход — одна строка или список строк. Перед векторизацией каждый текст проходит `clean_text`. Выход — список имён `negative`, `neutral`, `positive` в том же порядке. Вероятностей нет: у LinearSVC их не бывает.

## Метрики

Числа ниже получены на отложенном test модели, обученной на 80% данных. Файл в `models/` обучен на всей подготовленной таблице, поэтому его веса с этим test-прогоном не совпадают.

| Вариант | Macro F1 | Recall negative |
| --- | ---: | ---: |
| Baseline: TF-IDF 1–2 + логистическая регрессия | 0.6832 | 0.2899 |
| TF-IDF 1–2 + LogReg, class_weight=balanced | 0.7103 | 0.6377 |
| TF-IDF 1–3 + LogReg | 0.6900 | 0.2899 |
| TF-IDF 1–2 + LinearSVC | 0.7333 | 0.4928 |

Лучший вариант по macro F1 — TF-IDF на n-граммах 1–2 и LinearSVC.

## Baseline, лучший вариант и ошибки

Baseline — логистическая регрессия без весов классов. Accuracy на test 0.7992 выше macro F1, потому что `neutral` занимает большую часть выборки. Класс `negative` модель почти не находит: recall 0.2899.

Веса классов поднимают recall `negative` до 0.6377, но accuracy падает до 0.7815, и macro F1 остаётся 0.7103. Триграммы добавляют 0.0067 к macro F1 и recall `negative` не меняют. LinearSVC даёт macro F1 0.7333 и recall `negative` 0.4928, accuracy при этом 0.8054.

На test у LinearSVC 187 ошибок из 961. 137 из них — граница `neutral` и `positive`. `negative` чаще уходит в `positive` (22), чем в `neutral` (13). Подробности и примеры — в `reports/day05_error_analysis.md`.

## Demo

Фраз нет в CSV. Классы — фактический ответ сохранённой модели.

| Текст | Предсказание |
| --- | --- |
| Apple shares jump 5% after a record earnings report | positive |
| The company reported a quarterly loss and announced mass layoffs | positive |
| The board will meet on Thursday to discuss the agenda | neutral |
| The stock won't stay down | positive |
| $AAPL Sales -4% | positive |

Фраза про убыток и увольнения тоже получает `positive`. «won't stay down» и «Sales -4%» взяты из тех же ситуаций, что и ошибки разбора.

## Ограничения

Модель рассчитана на короткие английские финансовые тексты. Мешок слов не держит отрицание, смешанную полярность и падение, заданное числом вроде `-4%`. После удаления конфликтов класс `negative` остаётся маленьким, поэтому на нём качество ниже, чем на `neutral` и `positive`.

Macro F1 0.7333 измерен на модели, обученной на 80% строк. Сохранённые `vectorizer.joblib` и `model.joblib` обучены на всех 4803 строках. Test-отчёт к этим файлам не относится.
