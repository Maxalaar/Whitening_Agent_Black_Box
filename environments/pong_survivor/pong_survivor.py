import gymnasium as gym
import gymnasium.spaces
import numpy as np

from typing import TYPE_CHECKING, Any, SupportsFloat, TypeVar

from gymnasium.envs.registration import EnvSpec

from environments.pong_survivor.ball import Ball, ball_observation_space
from environments.pong_survivor.paddle import Paddle, paddle_observation_space
from environments.pong_survivor.render import RenderEnvironment

if TYPE_CHECKING:
    pass

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")


class PongSurvivor(gym.Env):
    def __init__(self, environment_configuration: dict):
        self.balls = []
        self.paddles = []

        self.time_step: float = 0.02
        self.max_time: float = 50
        self.spec = EnvSpec('PongSurvivor')
        self.spec.max_episode_steps = int(self.max_time / self.time_step)

        self.play_area: np.ndarray = np.array([100, 100])

        for i in range(environment_configuration.get('number_ball', 1)):
            self.balls.append(Ball(self, i))
        for i in range(environment_configuration.get('number_paddle', 1)):
            self.paddles.append(Paddle(self, i))

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = self._get_observation_space()

        self.render_mode = environment_configuration.get('render_mode', None)
        self.render_environment = None

        self._current_time_steps: int = None
        self.terminated = None
        self.truncated = None

        self.reset()

    def reset(self, *, seed=None, options=None):
        self._current_time_steps = 0
        self.terminated = False
        self.truncated = False

        for ball in self.balls:
            ball.reset()

        for paddle in self.paddles:
            paddle.reset()

        return self._get_observation(), {}

    def step(self, action):
        self._current_time_steps += 1

        for ball in self.balls:
            ball.move(self.time_step)

        self.paddles[0].move(action, self.time_step)

        if self._current_time_steps > self.spec.max_episode_steps:
            self.terminated = True

        return self._get_observation(), 1.0/self.spec.max_episode_steps, self.terminated, self.truncated, {}

    def render(self):
        if self.render_environment is None:
            self.render_environment = RenderEnvironment(self)

        return self.render_environment.render()

    def _get_observation_space(self):
        observation_space = {}

        for ball in self.balls:
            observation_space[ball.id] = ball_observation_space()

        for paddle in self.paddles:
            observation_space[paddle.id] = paddle_observation_space()

        return gymnasium.spaces.Dict(observation_space)

    def _get_observation(self):
        observation = {}

        for ball in self.balls:
            observation[ball.id] = ball.observation()

        for paddle in self.paddles:
            observation[paddle.id] = paddle.observation()

        return observation

