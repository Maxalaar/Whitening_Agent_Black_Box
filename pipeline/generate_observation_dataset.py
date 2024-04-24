import gymnasium
import numpy as np
import ray
import cv2
from ray.rllib.algorithms import PPO, Algorithm, AlgorithmConfig
from ray.tune import Tuner
import gymnasium as gym
from utilities.dataset_handler import DatasetHandler
from utilities.dictionary_merger import dictionary_merger
from utilities.global_include import get_workers
import torch
import dask.array as da


def post_rendering_processing(images):
    new_width = 246
    ratio = images.shape[1] / images.shape[0]
    new_height = int(new_width / ratio)
    resize_images = cv2.resize(images, (new_width, new_height))
    gray_images = cv2.cvtColor(resize_images, cv2.COLOR_RGB2GRAY)

    return gray_images


def harvest(policy, number_episode, environment_configuration, environment_creator):
    observations = []
    renderings = []
    information: dict = None
    index_episodes = []

    for i in range(number_episode):
        index_start_episode = len(observations)

        environment_configuration['render_mode'] = 'rgb_array'
        environment: gym.Env = environment_creator(environment_configuration)
        observation, info = environment.reset()
        observation = gymnasium.spaces.utils.flatten(environment.observation_space, observation)
        rendering = environment.render()
        observations.append(observation)
        renderings.append(post_rendering_processing(rendering))
        information = dictionary_merger(information, info)
        terminate = False
        truncated = False

        while not terminate and not truncated:
            action = policy.compute_actions(obs_batch=torch.tensor(np.array([observation])), explore=True)[0]
            observation, _, terminate, truncated, info = environment.step(action)
            observation = gymnasium.spaces.utils.flatten(environment.observation_space, observation)
            rendering = environment.render()

            observations.append(observation)
            renderings.append(post_rendering_processing(rendering))
            information = dictionary_merger(information, info)

        index_end_episode = len(observations)-1
        index_episodes.append(np.array([index_start_episode, index_end_episode]))

    observations = np.array(observations)
    renderings = np.array(renderings)
    index_episodes = np.array(index_episodes)

    data = {'observation': observations, 'rendering': renderings, 'index_episodes': index_episodes}

    for key in information.keys():
        value = information[key]
        data[key] = np.array(value)

    return data


def generate_observation_dataset(datasets_directory, rllib_trial_path, number_iteration, number_episode_per_worker):
    print('-- Generate observation dataset --')
    print()

    dataset_handler = DatasetHandler(datasets_directory, 'observation')

    tuner: Tuner = Tuner.restore(path=rllib_trial_path, trainable=PPO)
    result_grid = tuner.get_results()
    best_result = result_grid.get_best_result(metric='episode_reward_mean', mode='max')
    path_checkpoint: str = best_result.best_checkpoints[0][0].path
    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)

    environment_creator = algorithm.env_creator
    environment_configuration = algorithm.config.env_config

    workers = [worker for worker in get_workers(algorithm.workers)]

    for i in range(number_iteration):
        results = ray.get([worker.for_policy.remote(func=harvest, number_episode=number_episode_per_worker, environment_configuration=environment_configuration, environment_creator=environment_creator) for index, worker in enumerate(workers)])
        time_step_index = dataset_handler.size('observation')

        for key in results[0].keys():
            values = []
            for result in results:
                if key == 'index_episodes':
                    result[key] = result[key] + time_step_index
                    time_step_index = np.max(result[key]) + 1
                values.append(result[key])
            # value = np.concatenate(values, axis=0)
            value = da.concatenate(values)
            dataset_handler.save({key: value})

        del results
