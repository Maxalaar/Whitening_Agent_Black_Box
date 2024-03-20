import numpy as np
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

from utilities.dataset_handler import DatasetHandler
from utilities.global_include import datasets_directory, sklearn_directory
from utilities.sklearn_classifier_handler import SklearnClassifierHandler


class LatentSpaceVisualisation:
    def __init__(self, name: str):
        self.color_palette = matplotlib.colormaps['tab10']
        self.latent_space_dataset_handler: DatasetHandler = DatasetHandler(datasets_directory, name)

        self.latent_space_data = None
        self.action_data = None
        self.rendering_data = None

        self.tsne_latent_space = None
        self.latent_space_color = None

        self.kmeans_classifier = None
        self.latent_space_label = None
        self.cluster_labels = None
        self.cluster_average_rendering = []

    def load_data(self, number_data):
        data = self.latent_space_dataset_handler.load(['latent_space', 'action', 'rendering'], number_data=number_data)
        self.latent_space_data = data['latent_space']
        self.action_data = data['action']
        self.rendering_data = data['rendering']

    def load_kmeans_classifier(self, name):
        self.kmeans_classifier = SklearnClassifierHandler(sklearn_directory, name).load()

    def kmeans_classifier_predict(self):
        self.latent_space_label = self.kmeans_classifier.predict(self.latent_space_data)
        self.latent_space_color = [self.color_palette(label) for label in self.latent_space_label]
        self.cluster_labels = np.unique(self.latent_space_label)

    def compute_tsne(self):
        tsne = TSNE(n_components=2)
        self.tsne_latent_space = tsne.fit_transform(self.latent_space_data)

    def compute_average_rendering_cluster(self):
        for label in self.cluster_labels:
            index = np.where(self.latent_space_label == label)
            rendering = self.rendering_data[index]
            average_rendering = np.mean(rendering, axis=0)
            self.cluster_average_rendering.append(average_rendering)

    def plot(self):
        cloud_dots_figure, cloud_dots_axis = plt.subplots()
        cloud_dots_figure.canvas.manager.set_window_title('Latent Space')
        scatter = cloud_dots_axis.scatter(self.tsne_latent_space[:, 0], self.tsne_latent_space[:, 1], c=self.latent_space_color)

        figure_rendering, axis_rendering = plt.subplots()
        figure_rendering.canvas.manager.set_window_title('Environment')
        axis_rendering.axis('off')

        self.plot_cluster_average_rendering()

        def on_hover(event):
            if event.inaxes == cloud_dots_axis:
                contains, ind = scatter.contains(event)
                if contains:
                    index = ind['ind'][0]
                    axis_rendering.imshow(self.rendering_data[index], cmap='gray')
                    plt.draw()
                else:
                    axis_rendering.clear()
                    axis_rendering.axis('off')
                    plt.draw()

        cloud_dots_figure.canvas.mpl_connect('motion_notify_event', on_hover)
        plt.show()

    def plot_cluster_average_rendering(self):

        nb_images = len(self.cluster_average_rendering)
        nb_cols = 4
        nb_rows = (nb_images + nb_cols - 1) // nb_cols
        fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(6, 4))

        for i, ax in enumerate(axes.flat):
            if i < nb_images:
                ax.imshow(self.cluster_average_rendering[i] / 255.0, cmap='gray')
            ax.axis('off')

        # plt.tight_layout()


if __name__ == '__main__':
    latent_space_visualisation = LatentSpaceVisualisation('latent_space')
    latent_space_visualisation.load_data(100)
    latent_space_visualisation.load_kmeans_classifier('kmeans_classifier')
    latent_space_visualisation.kmeans_classifier_predict()
    latent_space_visualisation.compute_tsne()
    latent_space_visualisation.compute_average_rendering_cluster()
    latent_space_visualisation.plot()



