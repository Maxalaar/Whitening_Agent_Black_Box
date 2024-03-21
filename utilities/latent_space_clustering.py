from sklearn.cluster import KMeans

from utilities.dataset_handler import DatasetHandler
from utilities.sklearn_classifier_handler import SklearnClassifierHandler


def latent_space_clustering(datasets_directory, sklearn_directory, number_clusters, number_initialisation):
    latent_space_dataset_handler: DatasetHandler = DatasetHandler(datasets_directory, 'latent_space')
    kmeans_classifier_handler = SklearnClassifierHandler(sklearn_directory, 'kmeans_classifier')
    latent_space_dataset_handler.print_info()
    data = latent_space_dataset_handler.load(['latent_space'])

    latent_space_data = data['latent_space']

    kmeans_classifier = KMeans(n_clusters=number_clusters, n_init=number_initialisation)
    kmeans_classifier.fit(latent_space_data)
    kmeans_classifier_handler.save(kmeans_classifier)




