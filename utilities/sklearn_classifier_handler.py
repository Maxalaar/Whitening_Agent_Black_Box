from joblib import dump, load

from utilities.global_include import create_directory


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
