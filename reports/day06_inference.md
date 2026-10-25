# День 6. Предсказание на новых текстах

Сохранён победитель дня 4: TF-IDF на n-граммах 1–2 и LinearSVC. Оба файла получены из одного обучения на всей подготовленной таблице, 4803 строк. Это тот же алгоритм, что в днях 4–5, но другие веса: test-метрики описывают модель, обученную на 80% данных.

- Векторизатор: `models/vectorizer.joblib`
- Модель: `models/model.joblib`

У LinearSVC нет вероятностей классов. Ответ — имя класса.

## Формат

Вход `predict_sentiment`: одна строка или список строк на английском.

Перед векторизацией каждый текст проходит `clean_text`: нижний регистр, удаление HTML и ссылок, схлопывание пробелов. Векторизатор только преобразует текст (`transform`), словарь заново не строится.

Выход: список имён `negative`, `neutral`, `positive` в том же порядке, что и вход. Одна строка на входе тоже возвращает список из одного элемента.

## Примеры

Фразы новые, в CSV их нет. Предсказания получены загрузкой файлов с диска.

| Текст | После очистки | Предсказание |
| --- | --- | --- |
| Apple shares jump 5% after a record earnings report | apple shares jump 5% after a record earnings report | `positive` |
| The company reported a quarterly loss and announced mass layoffs | the company reported a quarterly loss and announced mass layoffs | `positive` |
| The board will meet on Thursday to discuss the agenda | the board will meet on thursday to discuss the agenda | `neutral` |
| The stock won't stay down | the stock won't stay down | `positive` |
| $AAPL Sales -4% | $aapl sales -4% | `positive` |

Фразы «won't stay down» и «Sales -4%» взяты из тех же ситуаций, что и ошибки дня 5. В таблице стоит фактический класс этой сохранённой модели.
