from common import *

from enum import IntEnum

from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns


class ClassType(IntEnum):
    """Enum that lists all classes of variable star types."""

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

    @classmethod
    def classify(cls, variable_star_type: str):
        """Preprocesses (combines into one) and returns the type of variable star."""

        if pd.isna(variable_star_type) or not isinstance(variable_star_type, str) or variable_star_type.strip() == "":
            return cls.UNKNOWN

        variable_star_type = variable_star_type.upper()

        ecl_markers = ["EA", "EB", "EW", "EC", "ELL", "E/RS", "E|", "E "]
        if any(m in variable_star_type for m in ecl_markers):
            return cls.ECLIPSING

        cep_markers = ["DCEP", "CW-FU", "CW", "CWA", "CWB", "RVA", "RV", "ACEP", "CEP"]
        if any(m in variable_star_type for m in cep_markers):
            return cls.CEPHEIDS

        rr_markers = ["RRAB", "RRC", "RRD", "RR"]
        if any(m in variable_star_type for m in rr_markers):
            return cls.RR_LYRAE

        short_puls = ["DSCT", "HADS", "SXPHE", "GDOR", "ROAP", "ROAM"]
        if any(m in variable_star_type for m in short_puls):
            return cls.DELTA_SCUTI_ETC

        lpv_markers = [" M ", "MIRA", "SR", "SRA", "SRB", "SRC", "SRD", "L ", "LB", "LC", "LPV"]
        if any(m in variable_star_type for m in lpv_markers):
            return cls.LONG_PERIOD

        rot_markers = ["BY", "RS", "ACV", "SPB", "ROT", "GCAS"]
        if any(m in variable_star_type for m in rot_markers):
            return cls.ROTATIONAL

        yso_markers = ["TTS", "EXOR", "UXOR", "INS", "IN", "INST", "CST"]
        if any(m in variable_star_type for m in yso_markers):
            return cls.ERUPTIVE

        cataclysmic_markers = ["UG", "NL", "AM", "ZAND", "IB", "ISB"]  # и др.
        if any(m in variable_star_type for m in cataclysmic_markers):
            return cls.CATACLYSMIC

        em_markers = ["WR", "BE", "FSCMA"]
        if any(m in variable_star_type for m in em_markers):
            return cls.EMISSION_WR

        return cls.UNKNOWN


BINARY_CLASSIFICATION_LABEL: str = "present"
MULTICLASS_CLASSIFICATION_LABEL: str = "class"

MULTICLASS_CLASSIFICATION_DATA: pd.DataFrame = pd.read_csv("datasets/multiclass_classification_stellar_data.csv")
MULTICLASS_CLASSIFICATION_DATA.dropna(subset=["min_mag", "max_mag"], inplace=True)
MULTICLASS_CLASSIFICATION_DATA[MULTICLASS_CLASSIFICATION_LABEL] = MULTICLASS_CLASSIFICATION_DATA["type"].apply(
    lambda vtype: int(ClassType.classify(vtype)))
MULTICLASS_CLASSIFICATION_DATA.drop(["type", BINARY_CLASSIFICATION_LABEL], axis=1, inplace=True)

# delete "YSO/ERUPTIVE", "EMISSION_WR", "UNKNOWN"
MULTICLASS_CLASSIFICATION_DATA = MULTICLASS_CLASSIFICATION_DATA[
    MULTICLASS_CLASSIFICATION_DATA[MULTICLASS_CLASSIFICATION_LABEL] != int(ClassType.ERUPTIVE)]
MULTICLASS_CLASSIFICATION_DATA = MULTICLASS_CLASSIFICATION_DATA[
    MULTICLASS_CLASSIFICATION_DATA[MULTICLASS_CLASSIFICATION_LABEL] != int(ClassType.EMISSION_WR)]
MULTICLASS_CLASSIFICATION_DATA = MULTICLASS_CLASSIFICATION_DATA[
    MULTICLASS_CLASSIFICATION_DATA[MULTICLASS_CLASSIFICATION_LABEL] != int(ClassType.UNKNOWN)]

MULTICLASS_CLASSIFICATION_X: pd.DataFrame = MULTICLASS_CLASSIFICATION_DATA.drop(MULTICLASS_CLASSIFICATION_LABEL, axis=1)
MULTICLASS_CLASSIFICATION_Y: pd.DataFrame = MULTICLASS_CLASSIFICATION_DATA[MULTICLASS_CLASSIFICATION_LABEL]


def plot_variable_ratio(save_plot=False) -> None:
    """Plots ratio between types of variable stars."""

    class_counts = MULTICLASS_CLASSIFICATION_Y.value_counts().sort_index()
    sns.barplot(x=[ClassType(c).name for c in class_counts.index], y=class_counts.values,
                palette=sns.hls_palette(8)[0:3:2])
    if save_plot:
        plt.savefig("multiclass_classification/docs/images/variable_ratio.png")
    plt.show()


def test_multiclass_classifier(classifier, x_test_data: t.Any, y_test_data: t.Any,
                               save_plot: t.Union[bool, str] = False):
    """Tests multiclass classifier by creating confusion matrix and recording metrics."""

    y_predicted = classifier.predict(x_test_data)

    print("Recorded metrics:\n", classification_report(y_test_data, y_predicted))

    cm = sns.heatmap(confusion_matrix(y_test_data, y_predicted), annot=True, fmt="d", cmap=plt.cm.Blues)
    cm.set(xlabel="Predicted class", ylabel="Real class")
    if save_plot:
        print()
        plt.savefig("binary_classification/docs/images/cm_" + str(type(classifier).__name__).lower() + (
            save_plot if isinstance(save_plot, str) else "") + ".png")
    plt.show()
