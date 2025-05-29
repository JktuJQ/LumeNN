from common import NNHyperparameters

import keras

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier

LOGISTIC_REGRESSION = LogisticRegression()
DEFAULT_RANDOM_FOREST = RandomForestClassifier()
CONFIGURED_RANDOM_FOREST = RandomForestClassifier(max_depth=11)
SGD = SGDClassifier(loss='modified_huber')
DEFAULT_GRADIENT_BOOSTING = GradientBoostingClassifier()
CONFIGURED_GRADIENT_BOOSTING = GradientBoostingClassifier(max_depth=13)


class NNClassifier:
    """Wrapper that represents binary classifier."""

    def __init__(self, hyperparameters: NNHyperparameters, classifier: keras.Sequential):
        self.hyperparameters = hyperparameters

        self.classifier = classifier
        self.classifier.compile(hyperparameters.optimizer, hyperparameters.loss)

    def predict(self, x_data):
        return self.classifier.predict(x_data)

    def fit(self, x_data, y_data):
        self.classifier.fit(x_data.to_numpy(), y_data.to_numpy(), epochs=self.hyperparameters.epochs)


NN_LOGISTIC_REGRESSION = NNClassifier(
    NNHyperparameters(5,
                      keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.ExponentialDecay(
                          1e-2,
                          decay_steps=15000,
                          decay_rate=0.01
                      )),
                      keras.losses.CategoricalFocalCrossentropy(gamma=1.0)
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
                      keras.losses.CategoricalFocalCrossentropy(gamma=1.0)
                      ),
    keras.Sequential([
        keras.layers.Dense(1024, keras.activations.mish),
        keras.layers.Dense(128, keras.activations.hard_shrink),
        keras.layers.Dense(1, keras.activations.sigmoid)
    ])
)
