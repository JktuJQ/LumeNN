from common import *

from enum import IntEnum

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


class ClassType(IntEnum):
    UNKNOWN = 0
    ECLIPSING = 1
    CEPHEIDS = 2
    RR_LYRAE = 3
    DELTA_SCUTI_ETC = 4
    LONG_PERIOD = 5
    ROTATIONAL = 6
    ERUPTIVE = 7
    CATACLYSMIC = 8
    EMISSION_WR = 9

    def __str__(self):
        if self == ClassType.UNKNOWN:
            return "UNKNOWN"
        elif self == ClassType.ECLIPSING:
            return "ECLIPSING"
        elif self == ClassType.CEPHEIDS:
            return "CEPHEIDS"
        elif self == ClassType.RR_LYRAE:
            return "RR_LYRAE"
        elif self == ClassType.DELTA_SCUTI_ETC:
            return "DELTA_SCUTI_ETC"
        elif self == ClassType.LONG_PERIOD:
            return "LONG_PERIOD"
        elif self == ClassType.ROTATIONAL:
            return "ROTATIONAL"
        elif self == ClassType.ERUPTIVE:
            return "ERUPTIVE"
        elif self == ClassType.CATACLYSMIC:
            return "CATACLYSMIC"
        elif self == ClassType.EMISSION_WR:
            return "EMISSION_WR"


def classify_type(vtype: str) -> ClassType:
    """
    Возвращает укрупнённый класс переменной звезды
    в зависимости от содержимого строки vtype.
    """

    if pd.isna(vtype) or not isinstance(vtype, str) or vtype.strip() == "":
        return ClassType.UNKNOWN  # Или "UNKNOWN", если хотите отдельно отметить NaN и пустые

    # Приведём к верхнему регистру для надёжного поиска подстрок
    t = vtype.upper()

    # --- 1) Затменные (Eclipsing Binaries) ---
    # Ищем любые подтипы: EA, EB, EW, EC, ELL, E/RS и т.д.
    # Включим также "E|" (встречается в комбинированных типах) и просто "E" (бывает)
    ecl_markers = ["EA", "EB", "EW", "EC", "ELL", "E/RS", "E|", "E "]
    if any(m in t for m in ecl_markers):
        return ClassType.ECLIPSING

    # --- 2) Цефеиды и родственные (DCEP, CW, RV Tauri, ACEP) ---
    cep_markers = ["DCEP", "CW-FU", "CW", "CWA", "CWB", "RVA", "RV", "ACEP", "CEP"]
    # Примечание: "RV" может пересекаться с "RVC", "ROT" и пр., поэтому в реальном коде можно уточнять условия, но здесь — упрощённо.
    if any(m in t for m in cep_markers):
        return ClassType.CEPHEIDS

    # --- 3) RR Лиры (RRAB, RRC, RRD, RR...) ---
    rr_markers = ["RRAB", "RRC", "RRD", "RR"]
    if any(m in t for m in rr_markers):
        return ClassType.RR_LYRAE

    # --- 4) Короткопериодические пульсаторы: DSCT, SXPHE, GDOR, roAp ---
    short_puls = ["DSCT", "HADS", "SXPHE", "GDOR", "ROAP", "ROAM"]
    if any(m in t for m in short_puls):
        return ClassType.DELTA_SCUTI_ETC

    # --- 5) Долгопериодические и полуправильные (M, SR, L) ---
    # Mira (M), SR, SRA, SRB, SRC, SRD, L, LB, LC, LPV
    lpv_markers = [" M ", "MIRA", "SR", "SRA", "SRB", "SRC", "SRD", "L ", "LB", "LC", "LPV"]
    # Для "M" можно проверять отдельно, чтобы не совпадало с "MISC", поэтому " M " с пробелами,
    # но тут для упрощения — любой "M". В реальном коде нужна аккуратность или RegEx.
    if any(m in t for m in lpv_markers):
        return ClassType.LONG_PERIOD

    # --- 6) Ротационные переменные (BY, RS, ACV, SPB, ROT, GCAS) ---
    rot_markers = ["BY", "RS", "ACV", "SPB", "ROT", "GCAS"]
    if any(m in t for m in rot_markers):
        return ClassType.ROTATIONAL

    # --- 7) Эруптивные/молодые звёзды (T Tauri, EXOR, UXOR, INS...) ---
    yso_markers = ["TTS", "EXOR", "UXOR", "INS", "IN", "INST", "CST"]
    # "CST" иногда "constant?", но бывает и у молодых/неясных
    if any(m in t for m in yso_markers):
        return ClassType.ERUPTIVE

    # --- Катаклизмические (UG, NL, AM, ZAND, IB, IS, ... ) ---
    cataclysmic_markers = ["UG", "NL", "AM", "ZAND", "IB", "ISB"]  # и др.
    if any(m in t for m in cataclysmic_markers):
        return ClassType.CATACLYSMIC

    # --- 9) Горячие эмиссионные/WR/Be/симбиотические (WR, BE, FSCMa...) ---
    em_markers = ["WR", "BE", "FSCMA"]
    if any(m in t for m in em_markers):
        return ClassType.EMISSION_WR

        # Если ничего не подошло, отправляем в "MISC" или "HYBRID"
    return ClassType.UNKNOWN


BINARY_CLASSIFICATION_LABEL: str = "present"
MULTICLASS_CLASSIFICATION_LABEL: str = "class"
MULTICLASS_CLASSIFICATION_DATA: pd.DataFrame = pd.read_csv("datasets/multiclass_classification_stellar_data.csv")
MULTICLASS_CLASSIFICATION_DATA.dropna(subset=["min_mag", "max_mag"], inplace=True)
MULTICLASS_CLASSIFICATION_DATA[MULTICLASS_CLASSIFICATION_LABEL] = MULTICLASS_CLASSIFICATION_DATA["type"].apply(
    lambda vtype: int(classify_type(vtype)))
MULTICLASS_CLASSIFICATION_DATA.drop("type", axis=1, inplace=True)

# delete "YSO/ERUPTIVE", "EMISSION_WR", "UNKNOWN"
MULTICLASS_CLASSIFICATION_DATA = MULTICLASS_CLASSIFICATION_DATA[
    MULTICLASS_CLASSIFICATION_DATA[MULTICLASS_CLASSIFICATION_LABEL] != "YSO/ERUPTIVE"]
MULTICLASS_CLASSIFICATION_DATA = MULTICLASS_CLASSIFICATION_DATA[
    MULTICLASS_CLASSIFICATION_DATA[MULTICLASS_CLASSIFICATION_LABEL] != "EMISSION_WR"]
MULTICLASS_CLASSIFICATION_DATA = MULTICLASS_CLASSIFICATION_DATA[
    MULTICLASS_CLASSIFICATION_DATA[MULTICLASS_CLASSIFICATION_LABEL] != "UNKNOWN"]

MULTICLASS_CLASSIFICATION_X: pd.DataFrame = MULTICLASS_CLASSIFICATION_DATA.drop(
    [MULTICLASS_CLASSIFICATION_LABEL, BINARY_CLASSIFICATION_LABEL], axis=1)
MULTICLASS_CLASSIFICATION_Y: pd.DataFrame = MULTICLASS_CLASSIFICATION_DATA[MULTICLASS_CLASSIFICATION_LABEL]

ERUPTIVE_KEYWORDS = [
    "FU", "GCAS", "IN", "RCB", "SDOR", "UV", "WR", "RS", "TTAU", "FLARE"
]

PULSATING_KEYWORDS = [
    "CEP", "CW", "DCEP", "DSCT", "M", "MIRA", "RR", "RV", "SR",
    "SRA", "SRB", "SRC", "SRD", "BCEP", "ZZ", "ACYG", "SXPHE", "PVTEL", "BLBOO"
]

ROTATING_KEYWORDS = [
    "ROT", "BY", "ELL", "ACV", "FKCOM", "SXARI", "PSR"
]

ECLIPSING_KEYWORDS = [
    "E", "EA", "EB", "EW", "EC", "EL", "ALGOL", "BLYR", "WUMA"
]


def plot_variable_ratio(data, save_plot=False) -> None:
    """Plots ratio between variable stars and non-variable stars."""

    sns.countplot(x=MULTICLASS_CLASSIFICATION_LABEL, data=data, palette=sns.hls_palette(8)[0:3:2])
    if save_plot:
        plt.savefig("binary_classification/docs/images/variable_ratio.png")
    plt.show()


def test_multiclass_classifier(classifier, x_test_data: t.Any, y_test_data: t.Any,
                               to_return=True, output=True, vmin=0):
    """Tests binary classifier by creating confusion matrix and recording metrics."""

    y_predicted = classifier.predict(x_test_data)

    print(classification_report(y_predicted, y_test_data, zero_division=0))
    ax = sns.heatmap(confusion_matrix(y_test_data, y_predicted), annot=True, vmax=y_predicted.size, vmin=vmin,
                     fmt="d", cmap=plt.cm.Blues)
    ax.set(xlabel="Predicted class", ylabel="Real class")
    plt.show()

    return {"Accuracy": accuracy_score(y_test_data, y_predicted),
            "Precision": precision_score(y_test_data, y_predicted, zero_division=0, average="weighted"),
            "Recall": recall_score(y_test_data, y_predicted, zero_division=0, average="weighted"),
            "F1": f1_score(y_test_data, y_predicted, zero_division=0, average="weighted")}
