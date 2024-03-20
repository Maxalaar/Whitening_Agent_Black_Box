from sklearn.cluster import KMeans

from utilities.dataset_handler import DatasetHandler
from utilities.global_include import datasets_directory, sklearn_directory, project_initialisation
from utilities.sklearn_classifier_handler import SklearnClassifierHandler

if __name__ == '__main__':
    project_initialisation()
    latent_space_dataset_handler: DatasetHandler = DatasetHandler(datasets_directory, 'latent_space')
    kmeans_classifier_handler = SklearnClassifierHandler(sklearn_directory, 'kmeans_classifier')
    latent_space_dataset_handler.print_info()
    data = latent_space_dataset_handler.load(['latent_space'])

    latent_space_data = data['latent_space']

    kmeans_classifier = KMeans(n_clusters=8, n_init=10)
    kmeans_classifier.fit(latent_space_data)
    kmeans_classifier_handler.save(kmeans_classifier)
    # cluster_assigned_label = kmeans_classifier.predict(latent_space_data)




