import typing as t

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import keras

from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Model
class NNHyperparameters:
    """`NNHyperparameters` holds all """

    def __init__(self, epochs: int, optimizer: t.Any, loss: t.Any):
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss


class Model(t.Protocol):
    """`Model` is an interface of any machine learning model."""

    def __init__(self, hyperparameters: NNHyperparameters, model: keras.Sequential):
        pass

    def predict(self, x_data: t.Any) -> t.Any:
        pass

    def fit(self, x_data: t.Any, y_data: t.Any):
        pass


# Data
def undersample(x_data: pd.DataFrame, y_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Undersamples datasets on the `y_data` parameter."""

    return RandomUnderSampler().fit_resample(x_data, y_data)


def train_test_split_data(x_data: pd.DataFrame, y_data: pd.DataFrame, train_test_ratio=0.3) \
        -> tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]:
    """Takes datasets and splits it into random train and test subsets."""

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=train_test_ratio)
    return (x_train, y_train), (x_test, y_test)


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
