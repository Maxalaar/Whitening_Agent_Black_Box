from joblib import dump, load
import os

from utilities.global_include import create_directory


def load_classifiers(directory_path):
    classifiers = {}

    for classifier_name in os.listdir(directory_path):
        classifier_path = os.path.join(directory_path, classifier_name)
        if os.path.isfile(classifier_path):
            classifiers[classifier_name] = SklearnClassifierHandler(directory_path, classifier_name).load()

    return classifiers


class SklearnClassifierHandler:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.classifier_path = self.path + '/' + self.name

    def save(self, classifier):
        create_directory(self.path)
        dump(classifier, self.classifier_path)

    def load(self):
        return load(self.classifier_path)
