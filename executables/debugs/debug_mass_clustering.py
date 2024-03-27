import dask.array as da
import numpy as np
from dask_ml.cluster import KMeans

from utilities.dataset_handler import DatasetHandler

if __name__ == '__main__':
    latent_space_dataset_handler: DatasetHandler = DatasetHandler(
        '/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/results/debug_experiment/datasets',
        'latent_space'
    )

    latent_space_dataset_handler.print_info()
    data = latent_space_dataset_handler.load(['latent_space'])
    data = data['latent_space']
    # data = np.array_split(data, 10)

    # Créer un modèle KMeans avec Dask-ML
    kmeans = KMeans(n_clusters=5)

    # Adapter le modèle aux données
    print('alo-1')
    kmeans.fit(data)
    print('alo+1')
    # Prévoir les clusters pour les données
    labels = kmeans.predict(data)

    # Afficher les centres des clusters
    print("Centers of clusters:", kmeans.cluster_centers_)

    # # Créer des données aléatoires (vous pouvez remplacer cela par vos propres données)
    # data = da.random.random((10000, 20), chunks=(1000, 20))
    #
    # # Créer un modèle KMeans avec Dask-ML
    # kmeans = KMeans(n_clusters=5)
    #
    # # Adapter le modèle aux données
    # kmeans.fit(data)
    #
    # # Prévoir les clusters pour les données
    # labels = kmeans.predict(data)
    #
    # # Afficher les centres des clusters
    # print("Centers of clusters:", kmeans.cluster_centers_)