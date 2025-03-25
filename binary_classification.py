from common import *

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier


LOGISTIC_REGRESSION = LogisticRegression(class_weight=CLASS_WEIGHTS)
DEFAULT_RANDOM_FOREST = RandomForestClassifier(class_weight=CLASS_WEIGHTS)
CONFIGURED_RANDOM_FOREST = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42,
                                                  class_weight=CLASS_WEIGHTS)
SGD = SGDClassifier(loss='modified_huber', class_weight=CLASS_WEIGHTS)
DEFAULT_GRADIENT_BOOSTING = GradientBoostingClassifier()
CONFIGURED_GRADIENT_BOOSTING = GradientBoostingClassifier(n_estimators=150, max_depth=15)

