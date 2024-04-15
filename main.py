import argparse
import os

from utilities.sklearn_classifier_handler import load_classifiers
from visualizations.visualization import Visualization
from utilities.dataset_handler import DatasetHandler

parser = argparse.ArgumentParser(description='Visualization of the results of the pipeline')
parser.add_argument('path', type=str, help='Path of the experiment results')
parser.add_argument('number_data', type=int, help='Amount of data to load for visualization')

visualization = Visualization('/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/static')
args = parser.parse_args()

path = os.path.abspath(args.path)

visualization.set_data(DatasetHandler(os.path.join(path, 'datasets'), 'latent_space').load(['latent_space', 'action', 'rendering'], number_data=args.number_data))
visualization.set_classifier(load_classifiers(os.path.join(path, 'sklearn')))

visualization.initialisation()
visualization.generation_renderings()
visualization.generate_clusters_representation()
visualization.display()
