import typing as t

import pandas as pd

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import keras

from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler


# Model
class NNHyperparameters:
    """`NNHyperparameters` holds all parameters of neural network."""

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
