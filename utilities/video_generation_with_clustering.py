import os

import cv2
import numpy as np

from pipeline.generate_videos import generate_video
from utilities.dataset_handler import DatasetHandler
from utilities.global_include import delete_directory, create_directory
from utilities.sklearn_classifier_handler import load_classifiers
from visualizations.visualization import finding_number_clusters


def video_generation_with_clustering():
    dataset = DatasetHandler('/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/results/debug/datasets', 'latent_space')
    episodes = dataset.load_episodes(keys=['rendering', 'latent_space'], number_episode=200)   # dataset.size('index_episodes')
    classifiers = load_classifiers('/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/results/debug/sklearn')
    classifier = classifiers[list(classifiers.keys())[0]]

    directory_path = './debug/video/'
    delete_directory(directory_path)
    create_directory(directory_path)

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
        generate_video(images=np.array(videos_frames[i]), output_video_path=directory_path+'cluster_'+str(i), fps=15)

