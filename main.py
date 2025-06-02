import sys
import typing as t

import common

import binary_classification.data as bd
import binary_classification.classifiers as bc

import multiclass_classification.data as md
import multiclass_classification.classifiers as mc

import keras


def main(argv: t.List[str]) -> None:
    while True:
        print()
        classification_type = input(
            "Do you want to try 'binary' ('b') or 'multiclass' ('m') classification?\n")

        if classification_type == "b":
            print()
            classifier_id = int(input(
                """`LumeNN` offers following classifiers for binary classification:
        `sklearn` models:
            'logistic regression' ('1'),
            'SVC' ('2'),
            'KNN' ('3'),
            'random forest' ('4'),
            'SGD' ('5'),
            'gradient boosting' ('6'),
            'stacking' ('7'),

        Neural networks:
            'neural network that emulates logistic regression' ('8'),
            'neural network classifier' ('9')\n"""))
            if classifier_id == 1:
                classifier = bc.LOGISTIC_REGRESSION_CLASSIFIER
            elif classifier_id == 2:
                classifier = bc.SVC_CLASSIFIER
            elif classifier_id == 3:
                classifier = bc.KNN_CLASSIFIER
            elif classifier_id == 4:
                classifier = bc.RANDOM_FOREST_CLASSIFIER
            elif classifier_id == 5:
                classifier = bc.SGD_CLASSIFIER
            elif classifier_id == 6:
                classifier = bc.GRADIENT_BOOSTING_CLASSIFIER
            elif classifier_id == 7:
                classifier = bc.STACKING_CLASSIFIER
            elif classifier_id == 8:
                classifier = bc.NN_LOGISTIC_REGRESSION_CLASSIFIER
            elif classifier_id == 9:
                classifier = bc.NN_CLASSIFIER
                classifier.classifier = keras.models.load_model("datasets/best_weights.keras")
            else:
                print("Wrong input - there is no classifier with code " + str(classifier_id))
                continue

            print()
            ratio = input("You can change ratio of train and test dataset or leave blank to use default.\n")
            train_data, test_data = common.train_test_split_data(bd.BINARY_CLASSIFICATION_X,
                                                                 bd.BINARY_CLASSIFICATION_Y,
                                                                 train_test_ratio=float(ratio) if ratio else 0.3)
            if classifier_id in (3, 6):
                train_data = common.undersample(*train_data)
            classifier.fit(*train_data)

            print()
            save_plot = input(
                "Do you want to save plot to file (write filename identifier or leave it blank otherwise)?\n")
            if save_plot:
                bd.test_binary_classifier(classifier, *test_data, save_plot=save_plot)
            else:
                bd.test_binary_classifier(classifier, *test_data)
            print()

        elif classification_type == "m":
            print()
            classifier_id = int(input(
                """`LumeNN` offers following classifiers for multiclass classification:
        `sklearn` models:
            'logistic regression' ('1'),
            'SVC' ('2'),
            'KNN' ('3'),
            'random forest' ('4'),
            'SGD' ('5'),
            'gradient boosting' ('6')\n"""))
            if classifier_id == 1:
                classifier = bc.LOGISTIC_REGRESSION_CLASSIFIER
            elif classifier_id == 2:
                classifier = bc.SVC_CLASSIFIER
            elif classifier_id == 3:
                classifier = bc.KNN_CLASSIFIER
            elif classifier_id == 4:
                classifier = bc.RANDOM_FOREST_CLASSIFIER
            elif classifier_id == 5:
                classifier = bc.SGD_CLASSIFIER
            elif classifier_id == 6:
                classifier = bc.GRADIENT_BOOSTING_CLASSIFIER
            else:
                print("Wrong input - there is no classifier with code " + str(classifier_id))
                continue

            print()
            ratio = input("You can change ratio of train and test dataset or leave blank to use default.\n")
            train_data, test_data = common.train_test_split_data(md.MULTICLASS_CLASSIFICATION_X,
                                                                 md.MULTICLASS_CLASSIFICATION_Y,
                                                                 train_test_ratio=float(ratio) if ratio else 0.3)

            classifier.fit(*train_data)

            print()
            save_plot = input(
                "Do you want to save plot to file (write filename identifier or leave it blank otherwise)?\n")
            if save_plot:
                md.test_multiclass_classifier(classifier, *test_data, save_plot=save_plot)
            else:
                md.test_multiclass_classifier(classifier, *test_data)
            print()

        else:
            print("There is no classification type with code ", classification_type)
            continue


if __name__ == "__main__":
    main(sys.argv)
