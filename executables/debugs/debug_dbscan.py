from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Génération de données de test
    X, _ = make_blobs(n_samples=1000, centers=4, cluster_std=0.6, random_state=0)

    # Création de l'objet DBSCAN
    epsilon = 0.35
    min_points = 5
    dbscan = DBSCAN(eps=epsilon, min_samples=min_points)

    # Apprentissage du modèle et prédiction des clusters
    labels = dbscan.fit_predict(X)

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("Nombre de clusters trouvés :", num_clusters)

    # Extraction des points de bruit
    noise_indices = labels == -1
    noise_points = X[noise_indices]

    # Affichage des clusters et des points de bruit
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10, alpha=0.7)
    plt.scatter(noise_points[:, 0], noise_points[:, 1], c='red', marker='x', label='Noise')
    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()