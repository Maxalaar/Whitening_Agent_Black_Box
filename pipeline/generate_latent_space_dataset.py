import numpy as np
import torch

import ray
from ray.rllib.algorithms import PPO, Algorithm
from ray.tune import Tuner
import dask.array as da

from utilities.dataset_handler import DatasetHandler
from utilities.global_include import get_workers


def harvest(policy, observations):
    return {
        'latent_space': policy.model.get_latent_space(torch.tensor(observations).to(policy.device)),
        'action': policy.compute_actions(obs_batch=observations, explore=False)[0],
    }


def generate_latent_space_dataset(datasets_directory, rllib_trial_path):
    print('-- Generate latent space dataset --')
    print()

    observations_dataset_handler = DatasetHandler(datasets_directory, 'observation')
    observations_dataset_handler.print_info()
    latent_space_dataset_handler = DatasetHandler(datasets_directory, 'latent_space')

    data = observations_dataset_handler.load(['observation', 'index_episodes', 'rendering'])
    data_observation = data['observation']

    tuner: Tuner = Tuner.restore(path=rllib_trial_path, trainable=PPO)
    result_grid = tuner.get_results()
    best_result = result_grid.get_best_result(metric='episode_reward_mean', mode='max')
    path_checkpoint: str = best_result.best_checkpoints[0][0].path
    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)

    data_chunks = np.array_split(data_observation, algorithm.workers.num_remote_workers())
    workers = [worker for worker in get_workers(algorithm.workers)]
    results = ray.get([worker.for_policy.remote(func=harvest, observations=data_chunks[index]) for index, worker in enumerate(workers)])

    for key in results[0].keys():
        values = []
        for result in results:
            values.append(np.array(result[key]))
        value = da.concatenate(values, axis=0)
        latent_space_dataset_handler.save({key: value})

    for subgroup_name in observations_dataset_handler.get_subgroups_name():
        subgroup_data = observations_dataset_handler.load([subgroup_name])
        latent_space_dataset_handler.save({subgroup_name: subgroup_data[subgroup_name]})
