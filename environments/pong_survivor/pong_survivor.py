import gymnasium as gym
import numpy as np

from typing import TYPE_CHECKING, Any, SupportsFloat, TypeVar

from utilities.dataset_handler import DatasetHandler

if TYPE_CHECKING:
    pass

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")


class PongSurvivor(gym.Env):
    def __init__(self, env_config):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0, 1, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.array([0, 0], dtype=np.float32), {}

    def step(self, action):
        return np.array([0, 0], dtype=np.float32), 1.0, True, False, {}

    def get_observation(self) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return np.array([0, 0], dtype=np.float32)
