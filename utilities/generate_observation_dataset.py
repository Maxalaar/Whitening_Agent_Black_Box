import numpy as np
import ray
import cv2
from ray.rllib.algorithms import PPO, Algorithm, AlgorithmConfig
from ray.tune import Tuner
import gymnasium as gym
from utilities.dataset_handler import DatasetHandler
from utilities.global_include import get_workers


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

    for i in range(number_episode):
        environment_configuration['render_mode'] = 'rgb_array'
        environment: gym.Env = environment_creator(environment_configuration)
        observation, _ = environment.reset()
        rendering = environment.render()
        observations.append(observation)
        renderings.append(post_rendering_processing(rendering))
        terminate = False

        while not terminate:
            action = policy.compute_actions(obs_batch=observation, explore=True)[0]
            observation, _, terminate, _, _ = environment.step(action)
            rendering = environment.render()
            observations.append(observation)
            renderings.append(post_rendering_processing(rendering))

    observations = np.array(observations)
    renderings = np.array(renderings)

    return {'observation': observations, 'rendering': renderings}


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

        for key in results[0].keys():
            values = []
            for result in results:
                values.append(result[key])
            value = np.concatenate(values, axis=0)
            dataset_handler.save({key: value})


















    #
    #
    # def harvest(policy, policy_id):
    #     observations = []
    #     renderings = []
    #
    #     for i in range(number_episode_per_worker):
    #         environment_configuration['render_mode'] = 'rgb_array'
    #         environment: gym.Env = environment_creator(environment_configuration)
    #         observation, _ = environment.reset()
    #         rendering = environment.render()
    #         observations.append(observation)
    #         renderings.append(post_rendering_processing(rendering))
    #         terminate = False
    #
    #         while not terminate:
    #             action = policy.compute_actions(obs_batch=observation, explore=True)[0]
    #             observation, _, terminate, _, _ = environment.step(action)
    #             rendering = environment.render()
    #             observations.append(observation)
    #             renderings.append(post_rendering_processing(rendering))
    #
    #     observations = np.array(observations)
    #     renderings = np.array(renderings)
    #
    #     return {'observation': observations, 'rendering': renderings}
    #
    # algorithm.workers._local_worker = None
    #
    # for _ in range(number_iteration):
    #     results = algorithm.workers.foreach_policy(harvest)
    #
    #     for key in results[0].keys():
    #         values = []
    #         for result in results:
    #             values.append(result[key])
    #         value = np.concatenate(values, axis=0)
    #         dataset_handler.save({key: value})
