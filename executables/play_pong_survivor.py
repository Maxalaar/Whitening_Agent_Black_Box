import pygame

from environments.pong_survivor.pong_survivor import PongSurvivor


if __name__ == '__main__':
    pygame.init()

    pong_survivor = PongSurvivor(environment_configuration={
        'render_mode': 'human',
        'frame_skip': 10,
        'number_ball': 1,
        'size_map_x': 100,
        'size_map_y': 100,
        'paddle_size': 15,
        'paddle_speed': 30,
        'ball_speed': 40,
    })

    running_episodes = True
    while running_episodes:
        pong_survivor.reset()
        running_episode = True
        action = 0
        total_reward = 0
        while running_episode:
            for event in pygame.event.get():
                touches = pygame.key.get_pressed()
                if touches[pygame.K_LEFT]:
                    action = 1
                if touches[pygame.K_RIGHT]:
                    action = 2
                if not touches[pygame.K_LEFT] and not touches[pygame.K_RIGHT]:
                    action = 0
                if event.type == pygame.QUIT:
                    running_episode = False
                    running_episodes = False

            observation, reward, terminated, truncated, info = pong_survivor.step(action)
            total_reward += reward

            if terminated or truncated:
                running_episode = False
                print('reward : ' + str(total_reward))

