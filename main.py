import sys
import typing as t

import common
import binary_classification as bc


def main(argv: t.List[str]) -> None:
    while True:
        print()
        classification_type = input(
            "Do you want to try 'binary' ('b') or 'multiclass' ('m') classification?\n")
        if classification_type == "b":
            print()
            classifier_id = int(input(
                """`LumeNN` offers following classifiers:
                'logistic regression' ('1'),
                'random forest with default parameters' ('2'),
                'random forest with configured parameters' ('3'),
                'SGD' ('4'),
                'gradient boosting with default parameters' ('5'),
                'gradient boosting with configured parameters' ('6'),
                'neural network configuration' ('7')\n"""))
            if classifier_id == 1:
                classifier = bc.LOGISTIC_REGRESSION
            elif classifier_id == 2:
                classifier = bc.DEFAULT_RANDOM_FOREST
            elif classifier_id == 3:
                classifier = bc.CONFIGURED_RANDOM_FOREST
            elif classifier_id == 4:
                classifier = bc.SGD
            elif classifier_id == 5:
                classifier = bc.DEFAULT_GRADIENT_BOOSTING
            elif classifier_id == 6:
                classifier = bc.CONFIGURED_GRADIENT_BOOSTING
            elif classifier_id == 7:
                classifier = bc.NN_MODEL
            else:
                print("Wrong input - there is no classifier with code " + str(classifier_id))
                continue

            print()
            ratio = input("You can change ratio of train and test dataset or leave blank to use default.\n")
            train_data, test_data = common.train_test_split_data(common.BINARY_CLASSIFICATION_X,
                                                                 common.BINARY_CLASSIFICATION_Y,
                                                                 train_test_ratio=float(ratio) if ratio else 0.3)
            if classifier_id in (5, 6, 7):
                train_data = common.undersample(*train_data)
            classifier.fit(*train_data)

            print()
            save_plot = input(
                "Do you want to save plot to file (write filename identifier or leave it blank otherwise)?\n")
            if save_plot:
                recorded_metrics = common.test_binary_classifier(classifier, *test_data, save_plot=save_plot)
            else:
                recorded_metrics = common.test_binary_classifier(classifier, *test_data)
            print()
            print("Recorded metrics: " + str(recorded_metrics))
        elif classification_type == "m":
            raise "Not yet implemented"
        else:
            print("Wrong input - there is no classification with code " + classification_type)
            continue


if __name__ == "__main__":
    main(sys.argv)
