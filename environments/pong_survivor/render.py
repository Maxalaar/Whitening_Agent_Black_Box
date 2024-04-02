import numpy as np
import pygame
import math

def calculate_angle(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)

def draw_arrow(surface, color, x1, y1, x2, y2, arrow_width):
    pygame.draw.line(surface, color, (x1, y1), (x2, y2))
    angle = calculate_angle(x1, y1, x2, y2)
    pygame.draw.polygon(surface, color, [(x2, y2),  (x2 - arrow_width * math.cos(angle - math.pi/6), y2 - arrow_width * math.sin(angle - math.pi/6)), (x2 - arrow_width * math.cos(angle + math.pi/6), y2 - arrow_width * math.sin(angle + math.pi/6))])

class RenderEnvironment:
    def __init__(self, environment):
        self.environment = environment
        self.window_width = 450
        self.window_size_coefficient = self.window_width / environment.play_area[0]
        self.window_size = (int(environment.play_area[0] * self.window_size_coefficient), int(environment.play_area[1] * self.window_size_coefficient))

        self.ball_size = int(2 * self.window_size_coefficient)
        self.arrow_size = 20
        self.paddle_offset = 3

        self.human_render_is_init: bool = False
        self.window = None
        self.clock = None

        self.canvas = pygame.Surface((self.window_size[0], self.window_size[0]))

    def render(self):
        self._render_frame()

        if self.environment.render_mode == 'rgb_array':
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))

        elif self.environment.render_mode == 'human':
            if not self.human_render_is_init:
                if not pygame.get_init():
                    pygame.init()
                    pygame.display.init()
                self.window = pygame.display.set_mode(self.window_size)
                self.clock = pygame.time.Clock()
                pygame.display.set_caption('Pong Survivor')
                self.human_render_is_init = True

            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(120)

    def _render_frame(self):
        self.canvas.fill((0, 0, 0))

        for ball in self.environment.balls:
            self._draw_ball(ball)

        for paddle in self.environment.paddles:
            self._draw_paddle(paddle)

    def _draw_ball(self, ball):
        pygame.draw.rect(
            self.canvas,
            (255, 255, 255),
            pygame.Rect(
                ball.position * self.window_size_coefficient - self.ball_size/2,
                (self.ball_size, self.ball_size),
            ),
        )

        if self.environment.display_arrows:
            velocity_norm = np.linalg.norm(ball.velocity)
            velocity = ball.velocity / velocity_norm * self.arrow_size
            draw_arrow(self.canvas, (255, 255, 255), ball.position[0] * self.window_size_coefficient, ball.position[1] * self.window_size_coefficient, ball.position[0] * self.window_size_coefficient + velocity[0], ball.position[1] * self.window_size_coefficient + velocity[1], 1)


    def _draw_paddle(self, paddle):
        pygame.draw.rect(
            self.canvas,
            (255, 0, 0),
            pygame.Rect(
                ((paddle.position[0] - paddle.size/2) * self.window_size_coefficient, (paddle.position[1]) * self.window_size_coefficient - self.paddle_offset),
                (paddle.size * self.window_size_coefficient, self.paddle_offset),
            ),
        )
