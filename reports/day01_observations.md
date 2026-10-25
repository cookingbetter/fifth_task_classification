# День 1. Наблюдения о данных

Задача — классификация тональности финансового текста на три класса: `negative`, `neutral`, `positive`. Строки на этом этапе не удаляются и не очищаются.

## Размер и схема

- Строк: 5842, колонок после загрузки: 2 (`text`, `sentiment`).
- Исходные имена колонок: `Sentence`, `Sentiment`.
- Типы: `text` — object, `sentiment` — object.
- Пропуски: `text` — 0, `sentiment` — 0.
- Пустых текстов: 0.

## Классы

| Класс | Количество | Доля | Медиана символов | Медиана слов |
| --- | ---: | ---: | ---: | ---: |
| negative | 860 | 14.7% | 94 | 17 |
| neutral | 3130 | 53.6% | 114 | 21 |
| positive | 1852 | 31.7% | 100 | 18 |

## Длины текстов

- Символы: медиана 107, среднее 117.0, минимум 9, максимум 315.
- Слова: медиана 19, среднее 21.0, минимум 2, максимум 81.
- Длины по классам близки: отдельный класс не выделяется длиной текста.

## Качество разметки и состав текстов

- Уникальных текстов: 5322.
- Дополнительных строк с уже встреченным текстом: 520.
- Текстов, у которых в данных больше одной метки: 514.
- Доля текстов с символом `$`: 15.5%.
- Доля текстов со ссылкой `http`: 4.3%.

Пример конфликта меток (`negative`, `neutral`):

> ( ADP News ) - Feb 11 , 2009 - Finnish wood products technology supplier Raute Oyj ( HEL : RUTAV ) said today its net profit decreased to EUR 4.7 million ( USD 6.1 m ) for 2008 from EUR 6.6 million for 2007 .

## Примеры

### negative

> $ESI on lows, down $1.50 to $2.50 BK a real possibility

> SSH COMMUNICATIONS SECURITY CORP STOCK EXCHANGE RELEASE OCTOBER 14 , 2008 AT 2:45 PM The Company updates its full year outlook and estimates its results to remain at loss for the full year .

### neutral

> According to the Finnish-Russian Chamber of Commerce , all the major construction companies of Finland are operating in Russia .

> ASSA ABLOY Kaupthing Bank gave a ` neutral ' recommendation and a share price target of 174 crowns $ 24.7 - 19 euro on Swedish lock maker Assa Abloy AB .

### positive

> The GeoSolutions technology will leverage Benefon 's GPS solutions by providing Location Based Search Technology , a Communities Platform , location relevant multimedia content and a new and powerful commercial model .

> $SPY wouldn't be surprised to see a green close

## Графики

- `reports/figures/sentiment_distribution.png`
- `reports/figures/text_length_distribution.png`
- `reports/figures/length_by_class.png`

## Выводы

Предсказывать нужно тональность короткого финансового текста. Три класса заданы явно, пропусков нет.

Классы несбалансированы: `neutral` — чуть больше половины выборки, `negative` — меньшинство, около 15%. Accuracy будет завышена за счёт нейтрального класса, поэтому на следующих этапах ориентир — macro F1.

Разметка шумная: сотни одинаковых текстов имеют сразу две метки, и во всех таких случаях это пара `negative` и `neutral`. Других конфликтов классов нет. Повторы на дне 1 оставлены: это наблюдение для подготовки данных, а не правка выборки.

В одной таблице смешаны два стиля. Есть развёрнутые новости о компаниях и короткие сообщения с тикерами и ссылками. Очистка должна сохранять оба типа, а не рассчитывать только на связный новостной абзац.

Тексты короткие: медиана около ста символов и меньше двадцати слов, длиннее 315 символов строк нет. Для baseline достаточно мешка слов или TF-IDF, длинный контекст здесь не требуется.
