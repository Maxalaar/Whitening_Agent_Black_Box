from sklearn.cluster import KMeans, DBSCAN

from utilities.dataset_handler import DatasetHandler
from utilities.sklearn_classifier_handler import SklearnClassifierHandler


def latent_space_clustering(datasets_directory, sklearn_directory):
    print('-- Latent Space Clustering --')
    print()
    latent_space_dataset_handler: DatasetHandler = DatasetHandler(datasets_directory, 'latent_space')
    kmeans_classifier_handler = SklearnClassifierHandler(sklearn_directory, 'kmeans_classifier')
    dbscan_classifier_handler = SklearnClassifierHandler(sklearn_directory, 'dbscan_classifier')
    latent_space_dataset_handler.print_info()
    data = latent_space_dataset_handler.load(['latent_space'])

    latent_space_data = data['latent_space']

    kmeans_classifier = KMeans(n_clusters=5, n_init=10)
    kmeans_classifier.fit(latent_space_data)
    kmeans_classifier_handler.save(kmeans_classifier)

    # data_dbscan = latent_space_data[:100000]
    # dbscan_classifier = DBSCAN(n_jobs=-1)
    # dbscan_classifier.fit(data_dbscan)
    # dbscan_classifier_handler.save(dbscan_classifier)




