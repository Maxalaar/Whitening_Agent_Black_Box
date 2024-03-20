from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

from utilities.dataset_handler import DatasetHandler
from utilities.global_include import datasets_directory, sklearn_directory
from utilities.sklearn_classifier_handler import SklearnClassifierHandler


def plot_latent_space(x, y, color, rendering):
    figure, axis = plt.subplots()
    scatter = axis.scatter(x, y, c=color)

    figure_rendering, axis_rendering = plt.subplots()
    axis_rendering.axis('off')

    def on_hover(event):
        if event.inaxes == axis:
            contains, ind = scatter.contains(event)
            if contains:
                index = ind['ind'][0]
                axis_rendering.clear()
                axis_rendering.imshow(rendering[index])
                plt.draw()
            else:
                axis_rendering.clear()
                plt.draw()

    figure.canvas.mpl_connect('motion_notify_event', on_hover)
    plt.show()


if __name__ == '__main__':
    latent_space_dataset_handler: DatasetHandler = DatasetHandler(datasets_directory, 'latent_space')
    latent_space_dataset_handler.print_info()
    data = latent_space_dataset_handler.load(['latent_space', 'action', 'rendering'], number_data=100)

    latent_space_dataset = data['latent_space']
    action_dataset = data['action']
    rendering_dataset = data['rendering']
    # color = np.where(action_dataset == 0, 'r', 'b')

    # pca = PCA(n_components=2)
    # vector_2d_pca = pca.fit_transform(latent_space_dataset)

    kmeans_classifier = SklearnClassifierHandler(sklearn_directory, 'kmeans_classifier').load()
    cluster_assigned_label = kmeans_classifier.predict(latent_space_dataset)

    # number_label = kmeans_classifier.n_clusters
    # color_palette = plt.cm.get_cmap('tab10', number_label)
    color_palette = matplotlib.colormaps['tab10']
    color = [color_palette(label) for label in cluster_assigned_label]

    tsne = TSNE(n_components=2)
    vector_2d_tsne = tsne.fit_transform(latent_space_dataset)

    # plt.figure(figsize=(12, 6))

    # plt.subplot(1, 2, 1)
    # plt.scatter(vector_2d_pca[:, 0], vector_2d_pca[:, 1], c=colors, marker='o', s=50)
    # plt.title('PCA')
    #
    # plt.subplot(1, 2, 2)
    # plt.scatter(vector_2d_tsne[:, 0], vector_2d_tsne[:, 1], c=colors, marker='o', s=50)
    # plt.title('t-SNE')

    # plt.subplot(1, 2, 2)
    # plt.scatter(latent_space_data[:, 0], latent_space_data[:, 1], c=colors, marker='o', s=50)
    # plt.title('Base')

    # plt.show()

    plot_latent_space(vector_2d_tsne[:, 0], vector_2d_tsne[:, 1], color, rendering_dataset)
