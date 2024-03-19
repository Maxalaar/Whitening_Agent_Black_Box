import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF1Policy

if __name__ == "__main__":
    worker = RolloutWorker(
      env_creator=lambda _: gym.make("CartPole-v1"),
      # default_policy_class=PPOTF1Policy,
      config=PPOConfig(),
    )
    print(worker.sample())