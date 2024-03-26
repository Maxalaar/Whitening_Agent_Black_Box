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


if __name__ == '__main__':
    ray.init(local_mode=False)
    project_initialisation()

    tuner: Tuner = Tuner.restore(path='/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/results/debug_experiment/rllib/rllib_base_trial', trainable=PPO)
    result_grid = tuner.get_results()
    best_result = result_grid.get_best_result(metric='episode_reward_mean', mode='max')
    path_checkpoint: str = best_result.best_checkpoints[0][0].path
    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)

    environment_creator = algorithm.env_creator
    environment_configuration = algorithm.config.env_config


    def post_rendering_processing(images):
        new_width = 246
        ratio = images.shape[1] / images.shape[0]
        new_height = int(new_width / ratio)
        resize_images = cv2.resize(images, (new_width, new_height))
        gray_images = cv2.cvtColor(resize_images, cv2.COLOR_RGB2GRAY)

        return gray_images

    def harvest(policy, policy_id):
        print('start harvest')
        observations = []
        renderings = []

        for i in range(100):
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

        print('stop harvest')
        return {'observation': observations, 'rendering': renderings}


    algorithm.workers._local_worker = None
    results = algorithm.workers.foreach_policy(harvest)

    for key in results[0].keys():
        values = []
        for result in results:
            values.append(result[key])
        value = np.concatenate(values, axis=0)
        print(value.shape)
        # dataset_handler.save({key: value})
