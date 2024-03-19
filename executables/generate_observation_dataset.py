import os

import numpy as np
import ray
from ray.rllib import Policy
from ray.rllib.algorithms import PPO, Algorithm
from ray.tune import Tuner
import gymnasium as gym

from utilities.dataset_handler import DatasetHandler
from utilities.global_include import project_initialisation, datasets_directory, rllib_directory


@ray.remote
class ObservationHarvestingWorker:
    def __init__(self, path_policy_storage):
        self.observations = None

        tuner: Tuner = Tuner.restore(path=path_policy_storage, trainable=PPO)
        result_grid = tuner.get_results()
        best_result = result_grid.get_best_result(metric='episode_reward_mean', mode='max')
        path_checkpoint: str = best_result.best_checkpoints[0][0].path
        algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)

        self.policy: Policy = algorithm.get_policy()
        self.environment_creator = algorithm.env_creator
        self.environment_configuration = algorithm.config.env_config

    def harvest(self):
        self.observations = []

        for i in range(10):
            environment: gym.Env = self.environment_creator(self.environment_configuration)
            observation, _ = environment.reset()
            self.observations.append(observation)
            terminate = False

            while not terminate:
                action = self.policy.compute_actions(obs_batch=observation, explore=True)[0]
                observation, _, terminate, _, _ = environment.step(action)
                self.observations.append(observation)

        post_processed_observations = self.post_processing_observations()
        self.observations = []

        return post_processed_observations

    def post_processing_observations(self):
        post_processed_observations = np.array(self.observations)
        return post_processed_observations


if __name__ == '__main__':
    ray.init(local_mode=False)
    project_initialisation()

    workers_number = 4

    dataset_handler = DatasetHandler(datasets_directory, 'observation')
    policy_storage_directory = os.path.join(rllib_directory, 'PPO_2024-03-15_17-08-53')
    dataset_handler.load()

    observation_harvesting_workers = [ObservationHarvestingWorker.remote(policy_storage_directory) for _ in range(workers_number)]

    results = ray.get([workers.harvest.remote() for workers in observation_harvesting_workers])
    results = np.concatenate(results, axis=0)
    dataset_handler.save({'observations': results})


    # model_id = load_model.remote(policy_storage_directory)

    # # Données d'inférence
    # data = ...
    #
    # # Parallélisez l'inférence
    # results_ids = [perform_inference.remote(ray.get(model_id), data_chunk) for data_chunk in data_chunks]
    #
    # # Récupérez les résultats
    # results = ray.get(results_ids)
    #
    # # Traitez les résultats
    # # ...
    #
    # # Arrêtez Ray

    ray.shutdown()
