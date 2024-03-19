import os
import numpy as np
import torch

import ray
from ray.rllib import Policy
from ray.rllib.algorithms import PPO, Algorithm
from ray.tune import Tuner
from ray.rllib.models import ModelV2

from utilities.dataset_handler import DatasetHandler
from utilities.global_include import project_initialisation, datasets_directory, rllib_directory


@ray.remote
class LatentSpaceHarvestingWorker:
    def __init__(self, path_policy_storage):
        self.observations = None

        tuner: Tuner = Tuner.restore(path=path_policy_storage, trainable=PPO)
        result_grid = tuner.get_results()
        best_result = result_grid.get_best_result(metric='episode_reward_mean', mode='max')
        path_checkpoint: str = best_result.best_checkpoints[0][0].path
        algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)
        policy: Policy = algorithm.get_policy()

        self.model: ModelV2 = policy.model

    def harvest(self, observations):
        return self.model.get_latent_space(torch.tensor(observations))


if __name__ == '__main__':
    ray.init(local_mode=True)
    project_initialisation()

    workers_number = 4

    policy_storage_directory = os.path.join(rllib_directory, 'PPO_2024-03-15_17-08-53')
    observations_dataset_handler = DatasetHandler(datasets_directory, 'observation')
    latent_space_dataset_handler = DatasetHandler(datasets_directory, 'latent_space')

    observations_dataset_handler.print_info()

    data_chunks = np.split(observations_dataset_handler.load('observation'), workers_number)
    latent_space_harvesting_workers = [LatentSpaceHarvestingWorker.remote(policy_storage_directory) for _ in range(workers_number)]

    results = ray.get([latent_space_harvesting_workers[index].harvest.remote(data_chunk) for index, data_chunk in enumerate(data_chunks)])
    results = np.concatenate(results, axis=0)

    latent_space_dataset_handler.save({'latents_spaces': results})

    ray.shutdown()
