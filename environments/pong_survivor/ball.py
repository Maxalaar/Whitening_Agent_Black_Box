import numpy as np


class Ball:
    def __init__(self, play_area: np.ndarray):
        self.play_area = play_area
        self.position: np.ndarray = np.random.uniform(low=np.array([0, 0]), high=self.play_area, size=2)
        self.speed: float = 1.0
        self.angle: float = np.random.uniform(0, 2*np.pi)
        self.velocity: np.ndarray = self._compute_velocity()

    def _compute_velocity(self):
        x = np.cos(self.angle) * self.speed
        y = np.sin(self.angle) * self.speed
        return np.array([x, y])

    def move(self, time_step):
        self.position += self.velocity * time_step

        if self.play_area[0] < self.position[0] < 0.0:
            pass


