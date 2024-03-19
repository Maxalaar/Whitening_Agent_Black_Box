import ray
from ray.rllib.algorithms.ppo import PPOConfig

import architectures.register_model
from environments.pong_survivor.pong_survivor import PongSurvivor

ray.init(local_mode=True)

config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment(env=PongSurvivor)
    .rollouts(num_rollout_workers=2)
    .resources(num_gpus=1)
    .framework('torch')
    .training(model={'custom_model': 'minimal_latent_space_model'})
    .evaluation(evaluation_num_workers=1)
)

algo = config.build()  # 2. build the algorithm,

for _ in range(5):
    print(algo.train())  # 3. train it,

algo.evaluate()  # 4. and evaluate it.
algo.save('./results/agent/')

ray.shutdown()
