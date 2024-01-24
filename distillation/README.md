# esd-ddp
[Efficient SciDev. Homework -- DDP](https://drive.google.com/drive/folders/1LWLkut23ovI0oza4bYHRLvGuqnMrz5WJ)

## Описание подхода

Модель предобучается с knowledge distillation, а затем дообучается на задачах с бинарной классификацией COLA и SST2.

## Метрики

### COLA

На COLA дообученная модель всегда предсказывает класс 1 (recall = 1, precision = sum(labels) / len(labels), где labels - класс, 0 или 1).

| Метрика       | Distilled BERT     | bert_base_uncased|
| ------------- |:------------------:|:----------------:|
| recall        |        1.0         |        0.95      |
| precision     |        0.69        |        0.83      |
| f1            |        0.81        |        0.87      |

### SST2

| Метрика       | Distilled BERT     | bert_base_uncased|
| ------------- |:------------------:|:----------------:|
| recall        |        0.82        |        0.91      |
| precision     |        0.85        |        0.93      |
| f1            |        0.83        |        0.92      |


