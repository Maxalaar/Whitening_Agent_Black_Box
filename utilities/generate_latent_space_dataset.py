import numpy as np
import torch

import ray
from ray.rllib import Policy
from ray.rllib.algorithms import PPO, Algorithm
from ray.tune import Tuner
from ray.rllib.models import ModelV2

from utilities.dataset_handler import DatasetHandler


@ray.remote
class LatentSpaceHarvestingWorker:
    def __init__(self, rllib_trial_path):
        self.observations = None

        tuner: Tuner = Tuner.restore(path=rllib_trial_path, trainable=PPO)
        result_grid = tuner.get_results()
        best_result = result_grid.get_best_result(metric='episode_reward_mean', mode='max')
        path_checkpoint: str = best_result.best_checkpoints[0][0].path
        algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)

        self.policy: Policy = algorithm.get_policy()
        self.model: ModelV2 = self.policy.model

    def harvest(self, observations):
        return {'latent_space': self.model.get_latent_space(torch.tensor(observations)), 'action': self.policy.compute_actions(obs_batch=observations, explore=False)[0]}


def generate_latent_space_dataset(datasets_directory, rllib_trial_path, workers_number):
    rllib_trial_path = rllib_trial_path
    observations_dataset_handler = DatasetHandler(datasets_directory, 'observation')
    latent_space_dataset_handler = DatasetHandler(datasets_directory, 'latent_space')

    data = observations_dataset_handler.load(['observation', 'rendering'])
    data_observation = data['observation']
    data_rendering = data['rendering']

    data_chunks = np.array_split(data_observation, workers_number)
    latent_space_harvesting_workers = [LatentSpaceHarvestingWorker.remote(rllib_trial_path) for _ in range(workers_number)]

    results = ray.get([latent_space_harvesting_workers[index].harvest.remote(data_chunk) for index, data_chunk in enumerate(data_chunks)])

    for key in results[0].keys():
        values = []
        for result in results:
            values.append(result[key])
        value = np.concatenate(values, axis=0)
        latent_space_dataset_handler.save({key: value})

    latent_space_dataset_handler.save({'rendering': data_rendering})