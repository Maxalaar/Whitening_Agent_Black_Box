import math
import time

from ray.util.client import ray


import ray
ray.shutdown()
ray.init(num_cpus=10)
from ray.rllib.policy.torch_policy_v2 import Policy


# from ray.rllib.policy.torch_policy_v2 import Policy

# from architectures.register_architectures import register_architectures

# from utilities.global_include import project_initialisation
# from ray.tune.registry import register_env
if __name__ == '__main__':
    @ray.remote
    class Counter(object):
        def __init__(self):
            self.n = 0
            print('init')

        def increment(self):
            print('start')
            while True:
                math.factorial(100000)

        def read(self):
            return self.n


    # ray.shutdown()
    # ray.init(num_cpus=10)
    #
    # import ray.rllib


    # project_initialisation()
    # register_architectures()
    counters = [Counter.remote() for i in range(10)]
    [c.increment.remote() for c in counters]
    futures = [c.read.remote() for c in counters]
    print(ray.get(futures))
    ray.shutdown()