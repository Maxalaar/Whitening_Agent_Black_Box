# import ray
from ray.rllib.algorithms import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air, tune

# from environments.pong_survivor.pong_survivor import PongSurvivor

if __name__ == '__main__':
    # ray.shutdown()
    # ray.init(local_mode=False, num_cpus=10)

    config = (  # 1. Configure the algorithm,
        PPOConfig()
        .environment(env='CartPole-v1')
        .rollouts(num_rollout_workers=5, create_env_on_local_worker=False, num_envs_per_worker=2, remote_worker_envs=True)
        .resources(num_learner_workers=5)
        # .resources(num_gpus=1, num_cpus_per_worker=1, num_learner_workers=5, num_gpus_per_worker=0)
        .framework('torch')
        # .training(model={'custom_model': 'minimal_latent_space_model'})
        # .evaluation(evaluation_num_workers=1)
    )

    tuner = tune.Tuner(
        trainable=PPO,
        param_space=config,
        run_config=air.RunConfig(
            name='debug_tuner',
            storage_path='/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/results/',
            stop={
                'time_total_s': 60 * 2,
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
