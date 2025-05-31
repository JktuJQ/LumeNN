from common import NNHyperparameters
from binary_classification.data import CLASS_WEIGHTS

import keras

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier


LOGISTIC_REGRESSION_CLASSIFIER = LogisticRegression(class_weight="balanced")
SVC_CLASSIFIER = SVC(class_weight="balanced")
KNN_CLASSIFIER = KNeighborsClassifier()
RANDOM_FOREST_CLASSIFIER = RandomForestClassifier(max_depth=11, class_weight="balanced")
SGD_CLASSIFIER = SGDClassifier(loss='modified_huber', class_weight="balanced")
GRADIENT_BOOSTING_CLASSIFIER = GradientBoostingClassifier(max_depth=13)


class NNClassifier:
    """Wrapper that represents binary classifier."""

    def __init__(self, hyperparameters: NNHyperparameters, classifier: keras.Sequential):
        self.hyperparameters = hyperparameters

        self.classifier = classifier
        self.classifier.compile(hyperparameters.optimizer, hyperparameters.loss)

    def predict(self, x_data):
        return list(map(int, self.classifier.predict(x_data) >= 0.5))

    def fit(self, x_data, y_data):
        self.classifier.fit(x_data.to_numpy(), y_data.to_numpy(), epochs=self.hyperparameters.epochs)


NN_LOGISTIC_REGRESSION_CLASSIFIER = NNClassifier(
    NNHyperparameters(50,
                      keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.ExponentialDecay(
                          1e-2,
                          decay_steps=15000,
                          decay_rate=0.01
                      )),
                      keras.losses.BinaryFocalCrossentropy(
                          apply_class_balancing=True, alpha=(1 - CLASS_WEIGHTS[0] / CLASS_WEIGHTS[1]),
                          gamma=1.0
                      )
                      ),
    keras.Sequential([
        keras.layers.Dense(1, keras.activations.sigmoid)
    ])
)

NN_CLASSIFIER = NNClassifier(
    NNHyperparameters(16,
                      keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.ExponentialDecay(
                          1e-2,
                          decay_steps=15000,
                          decay_rate=0.01
                      )),
                      keras.losses.BinaryFocalCrossentropy(
                          apply_class_balancing=True, alpha=(1 - CLASS_WEIGHTS[0] / CLASS_WEIGHTS[1]),
                          gamma=1.0
                      )
                      ),
    keras.Sequential([
        keras.layers.Dense(1024, keras.activations.mish),
        keras.layers.Dense(128, keras.activations.hard_shrink),
        keras.layers.Dense(1, keras.activations.sigmoid)
    ])
)
