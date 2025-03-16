# LumeNN

**LumeNN** — это нейронная сеть, которая решает проблему бинарной и многоклассовой
классификации переменных звёзд.


# Бинарная классификация

## Набор данных

Набор данных получен посредством слияния каталогов [APASS](https://www.aavso.org/apass) 
и [GALEX](https://galex.stsci.edu/GR6/) в [X-Match](http://cdsxmatch.u-strasbg.fr/) и просеивания
через [VSX](https://www.aavso.org/vsx/).
Далее датасет был очищен:
- удалео 512 строк с ошибкой более 1,
- удалена колонка type, ибо она заполнена менее, чем на 10%,
- удалены 4 строки без данных о минимальной и максимальной магнитудах (min_mag, max_mag).

Но в полученных данных ещё есть проблема с разделением на классы:

![Разделение на классы в полученных данных](img/origin_data_present_diagram.png)

Мы эту проблему пробовали решать двумя разными способами:

- Взвешивание классов
- Увеличение (Oversampling) и уменьшение (Undersampling) выборки

## Решение с помощью встроенных в scikit-learn моделей

### Логистическая регрессия

![Confusion matrix](img/cm_log_reg_wc.png)

Accuracy:  0.6071964017991005 \
Precision:  0.1360071988687492 \
Recall:  0.5467700258397933 \
F1 score:  0.21782993617459337

### Случайный лес - default parameters

![Confusion matrix](img/cm_random_forest_wc.png)

Accuracy:  0.9359975184821382\
Precision:  0.8842337375964718\
Recall:  0.4144702842377261\
F1 score:  0.5643912737508796

### Случайный лес - max_deph = 5, random_state=42

![Confusion matrix](img/cm_random_forest2_wc.png)

Accuracy:  0.7240345344569095\
Precision:  0.25548931220008725\
**Recall:  0.889620253164557**\
F1 score:  0.39697243560777223

Большое значение метрики Recall говорит нам о том, что модель смогла выявить большую
часть переменных звёзд из всех реально переменных. Хотя процент ложных срабатываний
удручает.

### SGDClassifier

loss: modified_huber

![Confusion matrix](img/cm_SGD_wc.png)

Accuracy:  0.8872460321563356\
Precision:  0.24120603015075376\
Recall:  0.04860759493670886\
F1 score:  0.08091024020227561

### Gradient Boosting

Так как у градиентного бустинга в sklearn отсутствует возможность задать веса
классам, к выборке был применен undersampling.

#### default parameters

![Confusion matrix](img/cm_gb0_us.png)

Accuracy:  0.8162642816522773\
Precision:  0.3515713744793639\
Recall:  0.9350453172205438\
F1 score:  0.5110071546505228

#### max_depth = 10

![Confusion matrix](img/cm_gb1_us.png)

Accuracy:  0.8786124179289666\
Precision:  0.450624558095687\
Recall:  0.991187143597719\
F1 score:  0.6195722618276086

## Выводы

![Metrics](img/metrics.png)

Видно, что лучше всего по совокупности метрик показал себя градиентный бустинг.