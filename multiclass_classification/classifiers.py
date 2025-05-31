from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier


SVC_CLASSIFIER = SVC(class_weight="balanced")
KNN_CLASSIFIER = KNeighborsClassifier()
RANDOM_FOREST_CLASSIFIER = RandomForestClassifier(max_depth=11, class_weight="balanced")
SGD_CLASSIFIER = SGDClassifier(loss='modified_huber', class_weight="balanced")
GRADIENT_BOOSTING_CLASSIFIER = GradientBoostingClassifier(max_depth=13)
