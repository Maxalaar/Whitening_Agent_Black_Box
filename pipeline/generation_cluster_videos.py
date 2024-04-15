import os.path

import numpy as np

from utilities.generate_videos import generate_video
from utilities.dataset_handler import DatasetHandler
from utilities.global_include import delete_directory, create_directory
from utilities.sklearn_classifier_handler import load_classifiers
from visualizations.visualization import finding_number_clusters


def generation_cluster_videos(video_directory, datasets_directory, classifiers_directory):
    print('-- Generate cluster videos --')
    print()
    dataset = DatasetHandler(datasets_directory, 'latent_space')
    episodes = dataset.load_episodes(keys=['rendering', 'latent_space'], number_episode=200)   # dataset.size('index_episodes')
    classifiers = load_classifiers(classifiers_directory)
    delete_directory(video_directory)
    create_directory(video_directory)

    for classifier_name in classifiers.keys():
        classifier_directory = os.path.join(video_directory, str(classifier_name))
        create_directory(classifier_directory)
        classifier = classifiers[classifier_name]
        videos_frames = {}
        for i in range(finding_number_clusters(classifier)):
            videos_frames[i] = []

        for episode_number in range(len(episodes['latent_space'])):
            latent_space = episodes['latent_space'][episode_number]
            cluster = classifier.predict(latent_space)
            render = episodes['rendering'][episode_number]
            for i in range(latent_space.shape[0]):
                videos_frames[cluster[i]].append(render[i])

        for i in range(finding_number_clusters(classifier)):
            generate_video(images=np.array(videos_frames[i]), output_video_path=classifier_directory+'/cluster_'+str(i), fps=15)

