import numpy as np
import cv2
import ray
from ray.rllib import Policy
from ray.rllib.algorithms import PPO, Algorithm, AlgorithmConfig
from ray.tune import Tuner
import gymnasium as gym

from environments.pong_survivor.pong_survivor import PongSurvivor
from utilities.dataset_handler import DatasetHandler
from utilities.global_include import project_initialisation
import numpy as np
import torch

import ray
from ray.rllib import Policy
from ray.rllib.algorithms import PPO, Algorithm
from ray.tune import Tuner
from ray.rllib.models import ModelV2

from utilities.dataset_handler import DatasetHandler


def minimal_function(policy, observations):
    return policy.model.get_latent_space(torch.tensor(observations).to(policy.device))


def minimal_function_2(worker, observations):
    return worker.for_policy(minimal_function, 'ouiii')


def get_workers(workers):
    worker_manager = getattr(workers, '_WorkerSet__worker_manager')
    return list(worker_manager.actors().values())


if __name__ == '__main__':
    ray.init(local_mode=True)
    project_initialisation()

    tuner: Tuner = Tuner.restore(path='/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/results/debug_experiment/rllib/rllib_base_trial', trainable=PPO)
    result_grid = tuner.get_results()
    best_result = result_grid.get_best_result(metric='episode_reward_mean', mode='max')
    path_checkpoint: str = best_result.best_checkpoints[0][0].path
    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)

    observations_dataset_handler = DatasetHandler('/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/results/debug_experiment/datasets', 'observation')
    latent_space_dataset_handler = DatasetHandler('/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/results/debug_experiment/datasets', 'latent_space')

    data = observations_dataset_handler.load(['observation', 'rendering'])
    data_observation = data['observation']
    data_rendering = data['rendering']

    data_chunks = np.array_split(data_observation, algorithm.workers.num_remote_workers())
    # workers = [worker for worker in algorithm.workers.remote_workers()]
    # workers = [worker for worker in algorithm.workers.healthy_worker_ids()]
    workers = [worker for worker in get_workers(algorithm.workers)]

    results = ray.get([worker.for_policy.remote(func=minimal_function, observations=data_chunks[index]) for index, worker in enumerate(workers)])



