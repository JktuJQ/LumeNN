import typing as t

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Model
class Model(t.Protocol):
    def predict(self, x_data: t.Any) -> t.Any:
        pass

    def fit(self, x_data: t.Any, y_data: t.Any):
        pass


# Data
def train_test_split_data(x_data: pd.DataFrame, y_data: pd.DataFrame, train_test_ratio=0.3) \
        -> tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]:
    """Takes data and splits it into random train and test subsets."""
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=train_test_ratio)
    return (x_train, y_train), (x_test, y_test)


def undersample(x_data: pd.DataFrame, y_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Undersamples data on the `y_data` parameter."""
    return RandomUnderSampler().fit_resample(x_data, y_data)


# Binary classification data
BINARY_CLASSIFICATION_LABEL: str = "variable"

BINARY_CLASSIFICATION_DATA: pd.DataFrame = pd.read_csv("data/binary_classification_stellar_data.csv")
BINARY_CLASSIFICATION_X: pd.DataFrame = BINARY_CLASSIFICATION_DATA.drop(BINARY_CLASSIFICATION_LABEL, axis=1)
BINARY_CLASSIFICATION_Y: pd.DataFrame = BINARY_CLASSIFICATION_DATA[BINARY_CLASSIFICATION_LABEL]

((BINARY_CLASSIFICATION_X_TRAIN, BINARY_CLASSIFICATION_Y_TRAIN),
 (BINARY_CLASSIFICATION_X_TEST, BINARY_CLASSIFICATION_Y_TEST)) = \
    (BINARY_CLASSIFICATION_TRAIN_DATA, BINARY_CLASSIFICATION_TEST_DATA) = \
    train_test_split_data(BINARY_CLASSIFICATION_X, BINARY_CLASSIFICATION_Y)

CLASS_WEIGHTS = dict(zip([0, 1],
                         compute_class_weight("balanced",
                                              classes=BINARY_CLASSIFICATION_Y.unique(),
                                              y=BINARY_CLASSIFICATION_Y)[::-1]))


def plot_variable_ratio(data, save_plot=False):
    """Plots ratio between variable stars and non-variable stars."""
    sns.countplot(x=BINARY_CLASSIFICATION_LABEL, data=data, palette='hls')
    plt.show()
    if save_plot:
        plt.savefig("binary_classification_variable_ratio.png")


def test_binary_classifier(classifier: Model, x_test_data: t.Any, y_test_data: t.Any, save_plot=False) \
        -> t.Dict[str, float]:
    """Tests binary classifier by creating confusion matrix and recording metrics."""
    y_predicted = classifier.predict(x_test_data)

    cm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_data, y_predicted), display_labels=[0, 1])
    cm.plot(cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.show()
    if save_plot:
        plt.savefig("confusion_matrix_" + type(classifier).__name__ + ".png")

    return record_metrics(y_test_data, y_predicted)


# Metrics
METRICS: t.Dict[str, t.Callable[[pd.DataFrame, pd.DataFrame], float]] = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "F1": f1_score
}


def record_metrics(y_true: pd.DataFrame, y_predicted: pd.DataFrame) -> t.Dict[str, float]:
    """Records all metrics that are in the `metrics` dictionary and returns dictionary with results."""
    return {metric_name: metric_fn(y_true, y_predicted) for (metric_name, metric_fn) in METRICS.items()}
