import os.path

import numpy as np
import matplotlib.pyplot as plt

from utilities.generate_videos import generate_video
from utilities.dataset_handler import DatasetHandler
from utilities.global_include import delete_directory, create_directory
from utilities.sklearn_classifier_handler import load_classifiers
from visualizations.visualization import finding_number_clusters


def generation_statistics_cluster(statistique_directory, datasets_directory, classifiers_directory):
    print('-- Generate statistics cluster --')
    print()
    dataset = DatasetHandler(datasets_directory, 'latent_space')
    dataset.print_info()
    episodes = dataset.load_episodes(keys=['latent_space', '_observation_paddle_0_position', '_observation_ball_0_position', 'action'], number_episode=50)   # dataset.size('index_episodes')
    classifiers = load_classifiers(classifiers_directory)
    delete_directory(statistique_directory)
    create_directory(statistique_directory)

    for classifier_name in classifiers.keys():
        classifier = classifiers[classifier_name]
        classifier_directory = os.path.join(statistique_directory, str(classifier_name))
        create_directory(classifier_directory)

        cluster_statistics = {}
        for i in range(finding_number_clusters(classifier)):
            cluster_statistics[i] = {
                '_observation_paddle_0_position': [],
                'action': [],
            }

        for episode_number in range(len(episodes['latent_space'])):
            latent_space = episodes['latent_space'][episode_number]
            paddle_position = episodes['_observation_paddle_0_position'][episode_number]
            action = episodes['action'][episode_number]

            cluster = classifier.predict(latent_space)
            for i in range(latent_space.shape[0]):
                cluster_statistics[cluster[i]]['_observation_paddle_0_position'].append(paddle_position[i])
                cluster_statistics[cluster[i]]['action'].append(action[i])

        for i in range(finding_number_clusters(classifier)):
            paddle_position = np.array(cluster_statistics[i]['_observation_paddle_0_position'])[:, 0]
            action = np.array(cluster_statistics[i]['action'])

            action_0 = paddle_position[action == 0]
            action_1 = paddle_position[action == 1]
            action_2 = paddle_position[action == 2]

            plt.hist(
                [action_0, action_1, action_2],
                bins=10,
                edgecolor='black',
                alpha=0.7,
                color=['red', 'green', 'blue'],
                label=['Idle', 'Left', 'Right'],
            )
            plt.xlabel('Paddle Position')
            plt.grid(True)
            plt.legend()
            plt.savefig(classifier_directory + f'/cluster_{i}')
            plt.clf()





