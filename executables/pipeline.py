import os
import ray
ray.shutdown()
ray.init(num_cpus=10)

from utilities.global_include import project_initialisation
from utilities.training_agent import training_agent
from utilities.generate_observation_dataset import generate_observation_dataset
from utilities.generate_latent_space_dataset import generate_latent_space_dataset
from utilities.latent_space_clustering import latent_space_clustering
from utilities.visualization import visualization


if __name__ == '__main__':
    experiment_name = 'debug_experiment'
    rllib_trial_name = 'rllib_base_trial'
    environment_name = 'CartPole-v1'
    architecture_name = 'minimal_latent_space_model'

    execution_directory = os.getcwd()
    results_directory = os.path.join(execution_directory, 'results')
    experiment_directory = os.path.join(results_directory, experiment_name)
    datasets_directory = os.path.join(experiment_directory, 'datasets')
    rllib_directory = os.path.join(experiment_directory, 'rllib')
    sklearn_directory = os.path.join(experiment_directory, 'sklearn')
    rllib_trial_path = os.path.join(rllib_directory, rllib_trial_name)

    project_initialisation()

    training_agent(
        rllib_directory=rllib_directory,
        rllib_trial_name=rllib_trial_name,
        environment_name=environment_name,
        architecture_name=architecture_name,
    )

    generate_observation_dataset(
        datasets_directory=datasets_directory,
        rllib_trial_path=rllib_trial_path,
        workers_number=5,
    )

    generate_latent_space_dataset(
        datasets_directory=datasets_directory,
        rllib_trial_path=rllib_trial_path,
        workers_number=5,
    )

    latent_space_clustering(
        datasets_directory=datasets_directory,
        sklearn_directory=sklearn_directory,
        number_clusters=5,
        number_initialisation=10,
    )

    visualization(
        datasets_directory=datasets_directory,
        sklearn_directory=sklearn_directory,
    )

    ray.shutdown()
