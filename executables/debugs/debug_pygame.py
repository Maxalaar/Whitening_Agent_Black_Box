import pygame
import random

# Définition des couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Paramètres de la fenêtre
WIDTH, HEIGHT = 800, 600

# Classe pour représenter une particule
class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 5
        self.color = RED
        self.speed_x = random.uniform(-1, 1)
        self.speed_y = random.uniform(-1, 1)

    def move(self):
        self.x += self.speed_x
        self.y += self.speed_y

        # Rebondir sur les bords de l'écran
        if self.x <= 0 or self.x >= WIDTH:
            self.speed_x *= -1
        if self.y <= 0 or self.y >= HEIGHT:
            self.speed_y *= -1

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

# Fonction principale
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simulation de particules")

    clock = pygame.time.Clock()

    particles = [Particle(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(50)]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)

        for particle in particles:
            particle.move()
            particle.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
