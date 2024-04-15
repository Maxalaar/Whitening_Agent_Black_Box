import os.path

import numpy as np

from utilities.generate_videos import generate_video
from utilities.dataset_handler import DatasetHandler
from utilities.global_include import delete_directory, create_directory
from utilities.sklearn_classifier_handler import load_classifiers
from visualizations.visualization import finding_number_clusters
import cv2


def display_numeric_value(image, value, position=(50, 200), color=(255, 255, 255), thickness=2, scale=1,
                          font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Displays a numeric value on a NumPy array image.

    Args:
        image (numpy.ndarray): The image on which to display the value.
        value (int or float): The numeric value to display.
        position (tuple): The coordinates of the text position (default: (50, 200)).
        color (tuple): The color of the text in BGR (default: (255, 255, 255)).
        thickness (int): The thickness of the text (default: 2).
        scale (float): The font scale (default: 1).
        font (int): The font type (default: cv2.FONT_HERSHEY_SIMPLEX).

    Returns:
        numpy.ndarray: The image with the numeric value displayed.
    """
    # Convert the value to a string
    text = str(value)

    # Draw the text on the image
    cv2.putText(image, text, position, font, scale, color, thickness)

    return image


def generation_clustered_episode_videos(video_directory, datasets_directory, classifiers_directory):
    print('-- Generate clustered episode videos --')
    print()
    number_episode = 200
    dataset = DatasetHandler(datasets_directory, 'latent_space')
    episodes = dataset.load_episodes(keys=['rendering', 'latent_space'], number_episode=number_episode)   # dataset.size('index_episodes')
    classifiers = load_classifiers(classifiers_directory)
    delete_directory(video_directory)
    create_directory(video_directory)

    for classifier_name in classifiers.keys():
        classifier_directory = os.path.join(video_directory, str(classifier_name))
        create_directory(classifier_directory)
        classifier = classifiers[classifier_name]

        for episode_number in range(number_episode):
            videos_frames = []
            latent_space = episodes['latent_space'][episode_number]
            timestep_number = latent_space.shape[0]
            cluster = classifier.predict(latent_space)
            render = episodes['rendering'][episode_number].copy()

            for i in range(timestep_number):
                image = display_numeric_value(render[i], cluster[i])
                videos_frames.append(image)

            generate_video(
                images=np.array(videos_frames),
                output_video_path=classifier_directory + '/video_' + str(episode_number),
                fps=15
            )