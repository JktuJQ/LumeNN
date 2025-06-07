import typing as t

import pandas as pd

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import keras

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
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
def oversample(x_data: pd.DataFrame, y_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Oversamples datasets on the `y_data` parameter using SMOTE method."""

    return SMOTE().fit_resample(x_data, y_data)


def undersample(x_data: pd.DataFrame, y_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Undersamples datasets on the `y_data` parameter."""

    return RandomUnderSampler().fit_resample(x_data, y_data)


def train_test_split_data(x_data: pd.DataFrame, y_data: pd.DataFrame, train_test_ratio=0.3) \
        -> tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]:
    """Takes datasets and splits it into random train and test subsets."""

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=train_test_ratio)
    return (x_train, y_train), (x_test, y_test)


def train_test_scale(train_data: tuple[pd.DataFrame, pd.DataFrame], test_data: tuple[pd.DataFrame, pd.DataFrame])\
        -> tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]:
    """Uses `StandardScaler` to scale train and test data."""

    x_train_data, y_train_data = train_data
    x_test_data, y_test_data = test_data

    scaler = StandardScaler()
    x_train_data = scaler.fit_transform(x_train_data)
    x_test_data = scaler.transform(x_test_data)
    return (x_train_data, y_train_data), (x_test_data, y_test_data)
