import ray
from ray import air, tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig, PPO

from utilities.global_include import rllib_directory, project_initialisation

if __name__ == '__main__':
    ray.init()
    project_initialisation()

    algorithm_configuration: AlgorithmConfig = (
        PPOConfig()
        .environment(env='pong_survivor')
        .framework('torch')
        .training(model={'custom_model': 'minimal_latent_space_model'})
    )

    tuner = tune.Tuner(
        trainable=PPO,
        param_space=algorithm_configuration,
        run_config=air.RunConfig(
            storage_path=rllib_directory,
            stop={
                'time_total_s': 20,
            },
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute='episode_reward_mean',
                checkpoint_score_order='max',
                # checkpoint_frequency=1,
                checkpoint_at_end=True,
            )
        ),
    )

    tuner.fit()

    ray.shutdown()
