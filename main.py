import sys
import typing as t

import common

import binary_classification.data as bd
import binary_classification.classifiers as bc

import keras


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
                'neural network that emulates logistic regression' ('7'),
                'neural network classifier' ('8')\n"""))
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
                classifier = bc.NN_LOGISTIC_REGRESSION
            elif classifier_id == 8:
                classifier = bc.NN_CLASSIFIER
                classifier.classifier = keras.models.load_model('datasets/best_weights.keras')
            else:
                print("Wrong input - there is no classifier with code " + str(classifier_id))
                continue

            print()
            ratio = input("You can change ratio of train and test dataset or leave blank to use default.\n")
            train_data, test_data = common.train_test_split_data(bd.BINARY_CLASSIFICATION_X,
                                                                 bd.BINARY_CLASSIFICATION_Y,
                                                                 train_test_ratio=float(ratio) if ratio else 0.3)
            if classifier_id in (5, 6):
                train_data = common.undersample(*train_data)
            classifier.fit(*train_data)

            print()
            save_plot = input(
                "Do you want to save plot to file (write filename identifier or leave it blank otherwise)?\n")
            if save_plot:
                recorded_metrics = bd.test_binary_classifier(classifier, *test_data, save_plot=save_plot)
            else:
                recorded_metrics = bd.test_binary_classifier(classifier, *test_data)

            print()
            record_metrics = input(
                "Do you want to record metrics in a file (write filename or leave it blank otherwise)?\n")
            if record_metrics:
                with open(record_metrics, mode="a") as f:
                    f.write(str(recorded_metrics) + "\n")
            print()
        elif classification_type == "m":
            raise "Not yet implemented"
        else:
            print("Wrong input - there is no classification with code " + classification_type)
            continue


if __name__ == "__main__":
    main(sys.argv)
