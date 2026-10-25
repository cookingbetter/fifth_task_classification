# День 3. Baseline

Цепочка: `text_clean` → TF-IDF → логистическая регрессия. TF-IDF: до 5000 признаков, n-граммы 1–2. Словарь векторизатора строится только по train. Логистическая регрессия: `max_iter=1000`, без весов классов.

Разбиение: test 20%, `random_state=42`, стратификация по `label`.

## Размеры

- Train: 3842 строк.
- Test: 961 строк.

| Класс | Train | Test |
| --- | ---: | ---: |
| negative | 274 | 69 |
| neutral | 2088 | 522 |
| positive | 1480 | 370 |

## Метрики на test

Точка отсчёта для следующих дней — macro F1: **0.6832**.

Accuracy: 0.7992. На этой выборке она выше macro F1, потому что `neutral` занимает большую часть теста.

| Класс | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| negative | 0.8696 | 0.2899 | 0.4348 | 69 |
| neutral | 0.8075 | 0.9080 | 0.8548 | 522 |
| positive | 0.7806 | 0.7405 | 0.7601 | 370 |

```
              precision    recall  f1-score   support

    negative     0.8696    0.2899    0.4348        69
     neutral     0.8075    0.9080    0.8548       522
    positive     0.7806    0.7405    0.7601       370

    accuracy                         0.7992       961
   macro avg     0.8192    0.6461    0.6832       961
weighted avg     0.8016    0.7992    0.7882       961
```

## Примеры предсказаний

По одному первому примеру каждого класса из test.

Истина: `negative`. Предсказание: `positive`.

```
$nq got hit hard lower this am --> looks like it has found support a 18.89
```

Истина: `neutral`. Предсказание: `neutral`.

```
vacon controls a further 5 % of the company via investment fund power fund i. eur 1.0 = usd 1.397
```

Истина: `positive`. Предсказание: `positive`.

```
demand was brisk as expected and order levels have remained high .
```

Модель в файл не сохраняется: это шаг дня 6.
