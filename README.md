# Финансовые новости: классификация тональности

Учебный pipeline классификации текстов. Готовы дни 1–6: загрузка, очистка, baseline, сравнение улучшений, разбор ошибок и предсказание новых текстов. Сборка единого запуска остаётся на день 7.

## Задача

По тексту финансовой новости или короткого рыночного сообщения предсказать тональность: `negative`, `neutral` или `positive`.

## Данные

Файл `data/data.csv`, 5842 строки. В исходном CSV колонки называются `Sentence` и `Sentiment`. Загрузчик приводит их к схеме `text` и `sentiment` и проверяет, что метки входят в три известных класса.

## Запуск

```bash
pip install -r requirements.txt
python eda.py
python preprocess.py
python train_baseline.py
python train_improved.py
python error_analysis.py
python predict.py
```

`python load_data.py` печатает ту же сводку без графиков и отчёта. `python preprocess.py` собирает таблицу для обучения: очищает текст, удаляет пустые строки, конфликты меток и повторы, кодирует классы. `python train_baseline.py` учит baseline и считает метрики на test. `python train_improved.py` сравнивает три улучшения с baseline на том же разбиении. `python error_analysis.py` строит матрицу ошибок победителя и разбирает примеры. `python predict.py` сохраняет модель и классифицирует новые тексты.

## Результаты дня 1

- `reports/day01_observations.md` — размер, пропуски, классы, длины, примеры и выводы.
- `reports/figures/sentiment_distribution.png` — распределение классов.
- `reports/figures/text_length_distribution.png` — гистограмма длины текста.
- `reports/figures/length_by_class.png` — длины по классам.

## Результаты дня 2

- `reports/day02_preprocessing.md` — правила очистки, сколько строк снято, классы до и после, примеры «было / стало».

Подготовленная таблица не сохраняется отдельным файлом. Следующие дни вызывают `prepare_dataset()` из `preprocess.py`, а новые тексты очищаются той же `clean_text()`.

## Результаты дня 3

- `reports/day03_baseline.md` — схема baseline, размеры train/test, метрики по классам и примеры предсказаний.
- `reports/baseline_result.txt` — macro F1 на test, точка отсчёта для сравнения.

Векторизатор учится только на train. Модель в файл пока не пишется.

## Результаты дня 4

- `reports/day04_comparison.md` — три улучшения против baseline, macro F1, recall класса `negative` и выбранный вариант.

Сравнение идёт на том же test, что и день 3. Файл модели пишет день 6.

## Результаты дня 5

- `reports/day05_error_analysis.md` — матрица ошибок LinearSVC, примеры по каждой паре классов и выводы.
- `reports/figures/confusion_matrix.png` — тепловая карта матрицы.

Разбор идёт на том же test, что и дни 3–4.

## Результаты дня 6

- `models/vectorizer.joblib` и `models/model.joblib` — TF-IDF и LinearSVC, обученные на всей подготовленной таблице.
- `reports/day06_inference.md` — формат входа и выхода и пять новых предсказаний.

Вызов: `predict_sentiment("текст")` или `predict_sentiment(["текст", "..."])`. Текст сначала очищается той же `clean_text()`. Ответ — список имён классов `negative`, `neutral`, `positive`. Вероятностей нет: у LinearSVC их не бывает.

## Дальше

1. День 1 — загрузка и первичный анализ.
2. День 2 — очистка текста и подготовка признаков.
3. День 3 — baseline-модель.
4. День 4 — улучшение baseline.
5. День 5 — анализ ошибок.
6. День 6 — сохранение модели и предсказание на новых текстах.
7. День 7 — сборка единого classification pipeline.
