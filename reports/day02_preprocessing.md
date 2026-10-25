# День 2. Предобработка

Тексты приводятся к одному виду функцией `clean_text`: нижний регистр, удаление HTML и ссылок `http://` / `https://`, схлопывание пробелов. Знаки `$` у тикеров и остальная пунктуация сохраняются.

В обучающую таблицу не попадают:

- пустой текст после очистки;
- один и тот же очищенный текст с разными метками, такие строки удаляются все;
- точные повторы очищенного текста, остаётся первая строка.

Метки кодируются так: `negative` = 0, `neutral` = 1, `positive` = 2. Исходное имя класса остаётся в колонке `sentiment`.

На очищенном тексте считаются `char_count`, `word_count` и `dollar_count`. Это ещё не вектор для модели.

## Сколько строк снято

- Было строк: 5842.
- Пустой текст: 0.
- Конфликт меток: 1028 строк (514 уникальных текстов).
- Повторы текста: 11.
- Осталось строк: 4803.

## Классы

| Класс | Код | Было | Конфликт | Повтор | Осталось |
| --- | ---: | ---: | ---: | ---: | ---: |
| negative | 0 | 860 | 514 | 3 | 343 |
| neutral | 1 | 3130 | 514 | 6 | 2610 |
| positive | 2 | 1852 | 0 | 2 | 1850 |

Класс `negative` сократился с 860 до 343. Из снятых строк этого класса 514 удалены из-за конфликта меток и 3 как повторы. Строки не возвращаются: у конфликта нет надёжной единственной метки.

## Примеры

### Новость

Класс: `positive`

Было:

```
The GeoSolutions technology will leverage Benefon 's GPS solutions by providing Location Based Search Technology , a Communities Platform , location relevant multimedia content and a new and powerful commercial model .
```

Стало:

```
the geosolutions technology will leverage benefon 's gps solutions by providing location based search technology , a communities platform , location relevant multimedia content and a new and powerful commercial model .
```

### Тикер

Класс: `negative`

Было:

```
$ESI on lows, down $1.50 to $2.50 BK a real possibility
```

Стало:

```
$esi on lows, down $1.50 to $2.50 bk a real possibility
```

### Ссылка

Класс: `negative`

Было:

```
$SAP Q1 disappoints as #software licenses down. Real problem? #Cloud growth trails $MSFT $ORCL $GOOG $CRM $ADBE https://t.co/jNDphllzq5
```

Стало:

```
$sap q1 disappoints as #software licenses down. real problem? #cloud growth trails $msft $orcl $goog $crm $adbe
```

### Конфликт меток

Метки одних и тех же данных: `negative`, `neutral`.

```
SSH COMMUNICATIONS SECURITY CORP STOCK EXCHANGE RELEASE OCTOBER 14 , 2008 AT 2:45 PM The Company updates its full year outlook and estimates its results to remain at loss for the full year .
```

После очистки:

```
ssh communications security corp stock exchange release october 14 , 2008 at 2:45 pm the company updates its full year outlook and estimates its results to remain at loss for the full year .
```

В подготовленной таблице этого текста нет.

## Колонки результата

`text`, `text_clean`, `sentiment`, `label`, `char_count`, `word_count`, `dollar_count`
