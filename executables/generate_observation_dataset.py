import os

import numpy as np
import cv2
import ray
from ray.rllib import Policy
from ray.rllib.algorithms import PPO, Algorithm
from ray.tune import Tuner
import gymnasium as gym

from utilities.dataset_handler import DatasetHandler
from utilities.global_include import project_initialisation, datasets_directory, rllib_directory


def post_rendering_processing(images):
    new_width = 246
    ratio = images.shape[1] / images.shape[0]
    new_height = int(new_width / ratio)
    resize_images = cv2.resize(images, (new_width, new_height))
    gray_images = cv2.cvtColor(resize_images, cv2.COLOR_RGB2GRAY)

    return gray_images


@ray.remote
class ObservationHarvestingWorker:
    def __init__(self, path_policy_storage):
        tuner: Tuner = Tuner.restore(path=path_policy_storage, trainable=PPO)
        result_grid = tuner.get_results()
        best_result = result_grid.get_best_result(metric='episode_reward_mean', mode='max')
        path_checkpoint: str = best_result.best_checkpoints[0][0].path
        algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)

        self.policy: Policy = algorithm.get_policy()
        self.environment_creator = algorithm.env_creator
        self.environment_configuration = algorithm.config.env_config

    def harvest(self):
        observations = []
        renderings = []

        for i in range(3):
            environment_configuration = self.environment_configuration
            environment_configuration['render_mode'] = 'rgb_array'
            environment: gym.Env = self.environment_creator(self.environment_configuration)
            observation, _ = environment.reset()
            rendering = environment.render()
            observations.append(observation)
            renderings.append(post_rendering_processing(rendering))
            terminate = False

            while not terminate:
                action = self.policy.compute_actions(obs_batch=observation, explore=True)[0]
                observation, _, terminate, _, _ = environment.step(action)
                rendering = environment.render()
                observations.append(observation)
                renderings.append(post_rendering_processing(rendering))

        observations = np.array(observations)
        renderings = np.array(renderings)

        return {'observation': observations, 'rendering': renderings}


if __name__ == '__main__':
    ray.init(local_mode=False)
    project_initialisation()

    workers_number = 4

    dataset_handler = DatasetHandler(datasets_directory, 'observation')
    policy_storage_directory = os.path.join(rllib_directory, 'PPO_2024-03-19_13-46-08')

    observation_harvesting_workers = [ObservationHarvestingWorker.remote(policy_storage_directory) for _ in range(workers_number)]

    results = ray.get([workers.harvest.remote() for workers in observation_harvesting_workers])

    for key in results[0].keys():
        values = []
        for result in results:
            values.append(result[key])
        value = np.concatenate(values, axis=0)
        dataset_handler.save({key: value})

    ray.shutdown()
