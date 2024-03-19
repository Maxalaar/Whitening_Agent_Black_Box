import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from utilities.dataset_handler import DatasetHandler
from utilities.global_include import datasets_directory

if __name__ == '__main__':
    latent_space_dataset_handler = DatasetHandler(datasets_directory, 'latent_space')
    latent_space_dataset = latent_space_dataset_handler.load('latents_spaces')

    pca = PCA(n_components=2)
    vector_2d_pca = pca.fit_transform(latent_space_dataset)

    tsne = TSNE(n_components=2)
    vector_2d_tsne = tsne.fit_transform(latent_space_dataset)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(vector_2d_pca[:, 0], vector_2d_pca[:, 1])
    plt.title('PCA')

    plt.subplot(1, 2, 2)
    plt.scatter(vector_2d_tsne[:, 0], vector_2d_tsne[:, 1])
    plt.title('t-SNE')

    plt.show()
