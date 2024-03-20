import os

from architectures.register_architectures import register_architectures
from environments.register_environments import register_environments
from utilities.environment_information import environment_information

execution_directory = os.getcwd()
results_directory = os.path.join(execution_directory, 'results')
datasets_directory = os.path.join(results_directory, 'datasets')
rllib_directory = os.path.join(results_directory, 'rllib')
sklearn_directory = os.path.join(results_directory, 'sklearn')


def create_directories():
    for path in [results_directory, datasets_directory, rllib_directory, sklearn_directory]:
        if not os.path.exists(path):
            os.makedirs(path)


def project_initialisation():
    environment_information()
    create_directories()
    register_environments()
    register_architectures()