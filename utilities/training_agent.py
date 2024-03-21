import os

from ray import air, tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig, PPO

from utilities.global_include import delete_directory


def training_agent(rllib_directory, rllib_trial_name, environment_name: str, architecture_name: str):
    delete_directory(os.path.expanduser('~/ray_results/' + rllib_trial_name))

    algorithm_configuration: AlgorithmConfig = (
        PPOConfig()
        .environment(env=environment_name)  # , disable_env_checking=True
        .framework('torch')
        .training(model={'custom_model': architecture_name})
    )

    tuner = tune.Tuner(
        trainable=PPO,
        param_space=algorithm_configuration,
        run_config=air.RunConfig(
            name=rllib_trial_name,
            storage_path=rllib_directory,
            stop={
                'time_total_s': 60 * 5,
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