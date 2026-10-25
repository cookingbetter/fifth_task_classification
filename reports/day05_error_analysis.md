# День 5. Разбор ошибок

Разбирается победитель дня 4: TF-IDF на n-граммах 1–2 и LinearSVC. Обучение только на train, разбиение то же, что в днях 3–4.

На test 961 строк, верных предсказаний 774, ошибок 187. Recall класса `negative`: 0.4928.

## Матрица ошибок

Строки — истинный класс, столбцы — предсказание. График: `reports/figures/confusion_matrix.png`.

| Истина \ предсказание | negative | neutral | positive | Всего |
| --- | ---: | ---: | ---: | ---: |
| negative | 34 | 13 | 22 | 69 |
| neutral | 4 | 460 | 58 | 522 |
| positive | 11 | 79 | 280 | 370 |

## Куда уходят ошибки

| Истина | Предсказание | Ошибок | Доля всех ошибок |
| --- | --- | ---: | ---: |
| positive | neutral | 79 | 42.2% |
| neutral | positive | 58 | 31.0% |
| negative | positive | 22 | 11.8% |
| negative | neutral | 13 | 7.0% |
| positive | negative | 11 | 5.9% |
| neutral | negative | 4 | 2.1% |

Граница `positive` и `neutral` даёт 137 из 187 ошибок. `negative` уходит в `positive` 22 раз и в `neutral` 13 раз. Обратных ошибок `neutral` → `negative` — 4.

## Примеры

Для каждой ошибочной пары взяты два первых текста test. Показан исходный текст.

### positive → neutral (79)

```
InterContinental Hotels Group share price climbs on $1.5bn special dividend
```

```
Japan's Asahi to submit bid next week for SABMiller's Grolsch and Peroni - Yomiuri
```

### neutral → positive (58)

```
Sales were down 14 per cent in Finland and up 27 per cent abroad where sales were lifted by the strong growth in sales in Russia and Lithuania .
```

```
To see a slide show of all the newest product releases from Fiskars .
```

### negative → positive (22)

```
$TWTR The best scenario going forward is this stock slowly falling everyday....Which is quite probable...
```

```
Berkshire discloses unit's ties to Iran, opens probe
```

### negative → neutral (13)

```
$GPS Gap September Comparable Store Sales -4%
```

```
Arm slammed by handset sales outlook
```

### positive → negative (11)

```
$JNK http://stks.co/2Fcm Does not want to go down... risk appetite still high
```

```
$FIO won't stay down
```

### neutral → negative (4)

```
This one was at one time targeted for June 5 , but if we had to guess , it 's been pushed back -- maybe we 're crazy , but we feel like it has n't leaked enough to meet that date at this point .
```

```
Sold some $TSLA Puts this morning Closed them at the close. Helps pay for monthly expenses
```

## Выводы

Основной объём ошибок — граница `neutral` и `positive`: 137 из 187. Позитивная новость без яркого оценочного слова уходит в `neutral`. В примерах это рост цены InterContinental на спецдивиденд и заявка Asahi на бренды SABMiller: в тексте есть climbs и bid, но нет прямого «хорошо» или «плохо». Нейтральная новость со смешанными числами уходит в `positive`: продажи упали на 14% в Финляндии и выросли на 27% за рубежом, а мешок слов цепляется за up, growth и lifted.

Отрицание и короткие тикерные сообщения ломают `negative`. «Does not want to go down» и «won't stay down» размечены как `positive`, а модель ставит `negative` из-за слова down. «$TWTR ... slowly falling» при этом предсказано как `positive`: рядом стоит best. «$GPS ... Sales -4%» уходит в `neutral`: падение задано числом, а не словом loss.

Дальше полезнее ловить отрицание, смешанную полярность и короткие тикерные сообщения, чем снова расширять n-граммы. Веса классов поднимают recall `negative`, но день 4 уже показал цену: падает accuracy. Новый эксперимент в этот день не запускается.

Модель в файл не сохраняется: это шаг дня 6.
