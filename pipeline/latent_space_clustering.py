from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from utilities.dataset_handler import DatasetHandler
from utilities.sklearn_classifier_handler import SklearnClassifierHandler


def auto_kmeans(range_cluster_number, data):
    inertia_values = []
    silhouette_scores = []

    for number_cluster in range_cluster_number:
        kmeans = KMeans(n_clusters=number_cluster, n_init=10)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)

        silhouette_avg = silhouette_score(data, kmeans.labels_, sample_size=2000)
        silhouette_scores.append(silhouette_avg)

    # print(inertia_values)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range_cluster_number, inertia_values, marker='o')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Somme des carrés intra-cluster')
    plt.title('Méthode du coude')

    plt.subplot(1, 2, 2)
    plt.plot(range_cluster_number, silhouette_scores, marker='o')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Coefficient de silhouette')
    plt.title('Coefficient de silhouette')

    plt.tight_layout()
    plt.show()


def latent_space_clustering(datasets_directory, sklearn_directory):
    print('-- Latent Space Clustering --')
    print()

    latent_space_dataset_handler: DatasetHandler = DatasetHandler(datasets_directory, 'latent_space')
    kmeans_classifier_handler = SklearnClassifierHandler(sklearn_directory, 'kmeans_classifier')
    dbscan_classifier_handler = SklearnClassifierHandler(sklearn_directory, 'dbscan_classifier')
    gaussian_mixture_handler = SklearnClassifierHandler(sklearn_directory, 'gaussian_mixture_classifier')
    mean_shift_handler = SklearnClassifierHandler(sklearn_directory, 'mean_shift_classifier')

    latent_space_dataset_handler.print_info()
    data = latent_space_dataset_handler.load(['latent_space'])

    latent_space_data = data['latent_space']

    kmeans_classifier = KMeans(n_clusters=10, n_init=10)
    kmeans_classifier.fit(latent_space_data)
    kmeans_classifier_handler.save(kmeans_classifier)

    # data_dbscan = latent_space_data
    # dbscan_classifier = DBSCAN(eps=0.40, n_jobs=-1)   # eps=0.4,
    # dbscan_classifier.fit(data_dbscan)
    # dbscan_classifier_handler.save(dbscan_classifier)

    # gaussian_mixture_classifier = GaussianMixture(n_components=3)
    # gaussian_mixture_classifier.fit(latent_space_data)
    # gaussian_mixture_handler.save(gaussian_mixture_classifier)

    # data_mean_shift = latent_space_data[:10000]
    # bandwidth = estimate_bandwidth(data_mean_shift, quantile=0.2, n_samples=500, n_jobs=-1)
    # mean_shift_classifier = MeanShift(bandwidth=bandwidth/2)
    # mean_shift_classifier.fit(data_mean_shift)
    # mean_shift_handler.save(mean_shift_classifier)

    # auto_kmeans(range_cluster_number=range(2, 10), data=latent_space_data)





