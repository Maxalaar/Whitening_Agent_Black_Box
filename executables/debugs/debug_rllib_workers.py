import gymnasium
import ray
from ray.rllib import Policy
from ray.rllib.algorithms import Algorithm
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF1Policy
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.tune import Tuner

from utilities.global_include import project_initialisation
import gymnasium as gym
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.algorithms.ppo import PPOConfig, PPO


class CustomWorker(RolloutWorker):
    def __init__(self, env_creator, policy, worker_index=0):
        pass
        # super().__init__(env_creator=env_creator)   # policy, worker_index
        # Ajoutez ici vos initialisations spécifiques au worker personnalisé

    def sample(self):
        # Remplacez cette méthode pour définir le comportement d'échantillonnage personnalisé
        pass

    def stop(self):
        # Remplacez cette méthode pour définir le comportement d'arrêt personnalisé
        pass


if __name__ == "__main__":
    ray.init()
    project_initialisation()

    # Créez un environnement ou une fonction de création d'environnement personnalisé
    def env_creator(env_config):
        return gymnasium.make('CartPole-v1')


    # Définissez votre politique ici, par exemple avec PPO
    # from ray.rllib.agents.ppo import PPOTrainer
    from ray.rllib.algorithms.ppo import PPOConfig

    config = {
        "env_config": {},  # Configuration de l'environnement
        "framework": "torch",  # Framework d'apprentissage
        # Autres configurations spécifiques à l'algorithme
    }

    # ppo_trainer = PPOTrainer(config=config, env=env_creator)

    rllib_trial_path = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/results/debug_experiment/rllib/rllib_base_trial/'
    tuner: Tuner = Tuner.restore(path=rllib_trial_path, trainable=PPO)
    result_grid = tuner.get_results()
    best_result = result_grid.get_best_result(metric='episode_reward_mean', mode='max')
    path_checkpoint: str = best_result.best_checkpoints[0][0].path
    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)

    policy: Policy = algorithm.get_policy()

    # Créez votre worker personnalisé
    custom_worker = CustomWorker(env_creator, policy)

    # Créez un ensemble de travailleurs (WorkerSet) pour gérer les workers
    workers = WorkerSet(
        env_creator, policy, custom_worker
    )

    # Exemple d'utilisation : effectuer des échantillonnages
    samples = workers.sample()
    print(samples)

    # N'oubliez pas d'arrêter l'environnement Ray après utilisation
    ray.shutdown()

    # project_initialisation()

    # rllib_trial_path = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/results/debug_experiment/rllib/rllib_base_trial/'
    # tuner: Tuner = Tuner.restore(path=rllib_trial_path, trainable=PPO)
    # result_grid = tuner.get_results()
    # best_result = result_grid.get_best_result(metric='episode_reward_mean', mode='max')
    # path_checkpoint: str = best_result.best_checkpoints[0][0].path
    # algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)
    #
    # policy: Policy = algorithm.get_policy()

    # worker = RolloutWorker(
    #     env_creator=lambda _: gym.make("CartPole-v1"),
    #     default_policy_class=PPOTorchPolicy)
    # print(worker.sample())
