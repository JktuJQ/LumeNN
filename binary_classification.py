from common import *

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier

import keras

LOGISTIC_REGRESSION = LogisticRegression(class_weight=CLASS_WEIGHTS)
DEFAULT_RANDOM_FOREST = RandomForestClassifier(class_weight=CLASS_WEIGHTS)
CONFIGURED_RANDOM_FOREST = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42,
                                                  class_weight=CLASS_WEIGHTS)
SGD = SGDClassifier(loss='modified_huber', class_weight=CLASS_WEIGHTS)
DEFAULT_GRADIENT_BOOSTING = GradientBoostingClassifier()
CONFIGURED_GRADIENT_BOOSTING = GradientBoostingClassifier(n_estimators=150, max_depth=15)


class NNClassifier:
    """Wrapper of `keras.Sequential` that represents binary classifier."""

    def __init__(self, classifier: keras.Sequential):
        self.classifier = classifier

    def predict(self, x_data):
        return self.classifier.predict(x_data) >= 0.5

    def fit(self, x_data, y_data):
        self.classifier.compile(NEURAL_NETWORK_OPTIMIZER, NEURAL_NETWORK_LOSS)
        self.classifier.fit(x_data.to_numpy(), y_data.to_numpy(), epochs=NEURAL_NETWORK_EPOCHS)


NN_MODEL = NNClassifier(keras.Sequential([
    keras.layers.Dense(512, keras.activations.leaky_relu),
    keras.layers.Dense(512, keras.activations.leaky_relu),
    keras.layers.Dense(1, keras.activations.sigmoid),
]))
