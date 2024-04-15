import gymnasium
import numpy as np
import ray
from ray.rllib.algorithms import PPO, Algorithm
from ray.tune import Tuner
import gymnasium as gym
from utilities.global_include import get_workers, create_directory, delete_directory
import torch

from utilities.generate_videos import generate_video, post_rendering_processing


def worker_generate_video_episodes(policy, worker_index, number_video, environment_configuration, environment_creator, video_directory):
    for i in range(number_video):
        observations = []
        renderings = []
        environment_configuration['render_mode'] = 'rgb_array'

        environment: gym.Env = environment_creator(environment_configuration)
        observation, _ = environment.reset()
        observation = gymnasium.spaces.utils.flatten(environment.observation_space, observation)
        rendering = environment.render()
        observations.append(observation)
        renderings.append(post_rendering_processing(rendering))
        terminate = False
        truncated = False

        while not terminate and not truncated:
            action = policy.compute_actions(obs_batch=torch.tensor(np.array([observation])), explore=True)[0]
            observation, _, terminate, truncated, _ = environment.step(action)
            observation = gymnasium.spaces.utils.flatten(environment.observation_space, observation)
            rendering = environment.render()
            observations.append(observation)
            renderings.append(post_rendering_processing(rendering))

        observations = np.array(observations)
        renderings = np.array(renderings)

        generate_video(images=renderings, output_video_path=video_directory + '/video_' + str(worker_index) + '-' + str(i))


def generate_video_episodes(video_directory, rllib_trial_path, number_video_per_worker):
    print('-- Generate episode videos --')
    print()
    delete_directory(video_directory)
    create_directory(video_directory)
    tuner: Tuner = Tuner.restore(path=rllib_trial_path, trainable=PPO)
    result_grid = tuner.get_results()
    best_result = result_grid.get_best_result(metric='episode_reward_mean', mode='max')
    path_checkpoint: str = best_result.best_checkpoints[0][0].path
    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)

    environment_creator = algorithm.env_creator
    environment_configuration = algorithm.config.env_config

    workers = [worker for worker in get_workers(algorithm.workers)]

    ray.get([worker.for_policy.remote(func=worker_generate_video_episodes, worker_index=index, number_video=number_video_per_worker, environment_configuration=environment_configuration, environment_creator=environment_creator, video_directory=video_directory) for index, worker in enumerate(workers)])
