from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier

LOGISTIC_REGRESSION_CLASSIFIER = LogisticRegression(class_weight="balanced")
SVC_CLASSIFIER = SVC(class_weight="balanced")
KNN_CLASSIFIER = KNeighborsClassifier()
RANDOM_FOREST_CLASSIFIER = RandomForestClassifier(max_depth=11, class_weight="balanced")
SGD_CLASSIFIER = SGDClassifier(loss='modified_huber', class_weight="balanced")
GRADIENT_BOOSTING_CLASSIFIER = GradientBoostingClassifier(max_depth=13)
STACKING_CLASSIFIER = StackingClassifier(estimators=[
    ("gradient boosting", GRADIENT_BOOSTING_CLASSIFIER),
    ("random forest", RANDOM_FOREST_CLASSIFIER)
], final_estimator=LogisticRegression())
MLP_CLASSIFIER = MLPClassifier(
    hidden_layer_sizes=(100, 50, 20, 10),
    activation="tanh",
    solver="adam",
    max_iter=250,
    learning_rate_init=0.005,
    batch_size=64
)
