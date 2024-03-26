import gymnasium
import ray
from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig

if __name__ == '__main__':
    ray.init(num_cpus=5)

    config = (  # 1. Configure the algorithm,
        PPOConfig()
        .environment("Taxi-v3")
        .rollouts(
            num_rollout_workers=4,
            remote_worker_envs=True,
            num_envs_per_worker=2,
        )
        .resources(
            num_learner_workers=2,
            num_cpus_per_worker=1,
            num_cpus_for_local_worker=1,
            num_cpus_per_learner_worker=1,
            num_gpus=0,
            num_gpus_per_worker=0,
            num_gpus_per_learner_worker=0,
        )
        .evaluation(evaluation_num_workers=1)
        .framework("torch")
        .training(
            model={"fcnet_hiddens": [64, 64]},
            train_batch_size=512*10,
            sgd_minibatch_size=64*10,
        )
    )

    algo = config.build()  # 2. build the algorithm,

    for _ in range(5):
        print(algo.train())  # 3. train it,

    algo.evaluate()  # 4. and evaluate it.