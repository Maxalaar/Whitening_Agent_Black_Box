from utilities.sklearn_classifier_handler import load_classifiers
from visualizations.visualization import Visualization
from utilities.dataset_handler import DatasetHandler

visualization = Visualization('/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/static')
visualization.set_data(DatasetHandler('/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/results/pong_big_experiment/datasets', 'latent_space').load(['latent_space', 'action', 'rendering'], number_data=50))
visualization.set_classifier(load_classifiers('/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/results/pong_big_experiment/sklearn'))

visualization.initialisation()
visualization.generation_renderings()
visualization.generate_clusters_representation()
visualization.display()
