from common import *

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

BINARY_CLASSIFICATION_LABEL: str = "variable"

BINARY_CLASSIFICATION_DATA: pd.DataFrame = pd.read_csv(
    "datasets/binary_classification_stellar_data.csv")
BINARY_CLASSIFICATION_DATA = BINARY_CLASSIFICATION_DATA.drop(["N"], axis=1)

BINARY_CLASSIFICATION_X: pd.DataFrame = BINARY_CLASSIFICATION_DATA.drop(BINARY_CLASSIFICATION_LABEL, axis=1)
BINARY_CLASSIFICATION_Y: pd.DataFrame = BINARY_CLASSIFICATION_DATA[BINARY_CLASSIFICATION_LABEL]

CLASS_WEIGHTS: t.Dict[int, float] = dict(zip([0, 1],
                                             compute_class_weight("balanced",
                                                                  classes=BINARY_CLASSIFICATION_Y.unique(),
                                                                  y=BINARY_CLASSIFICATION_Y)[::-1]))


def plot_variable_ratio(save_plot=False) -> None:
    """Plots ratio between variable stars and non-variable stars."""

    sns.countplot(x=BINARY_CLASSIFICATION_LABEL, y=BINARY_CLASSIFICATION_Y, palette=sns.hls_palette(8)[0:3:2])
    if save_plot:
        plt.savefig("binary_classification/docs/images/variable_ratio.png")
    plt.show()


def test_binary_classifier(classifier: Model, x_test_data: t.Any, y_test_data: t.Any, save_plot: t.Union[bool, str] = False):
    """Tests binary classifier by creating confusion matrix and recording metrics."""

    y_predicted = classifier.predict(x_test_data)

    print("Recorded metrics:\n", classification_report(y_test_data, y_predicted))

    cm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_data, y_predicted), display_labels=[0, 1])
    cm.plot(cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    if save_plot:
        print()
        plt.savefig("binary_classification/docs/images/cm_" + str(type(classifier).__name__).lower() + (
            save_plot if isinstance(save_plot, str) else "") + ".png")
    plt.show()
