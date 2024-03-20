import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Générer des données de points aléatoires
    # Générer des données de points aléatoires
    num_points = 50
    x = np.random.rand(num_points)
    y = np.random.rand(num_points)
    colors = np.random.rand(num_points, 3)  # Couleurs aléatoires pour chaque point

    # Créer une figure et un axe pour le nuage de points
    fig, ax = plt.subplots()

    # Afficher le nuage de points
    scatter = ax.scatter(x, y, c=colors)

    # Créer une nouvelle figure pour afficher l'image RGB
    fig_img, ax_img = plt.subplots()

    # Masquer les axes pour la figure de l'image
    ax_img.axis('off')

    # Préparer une image RGB aléatoire
    rgb_image = np.random.rand(100, 100, 3)

    # Afficher l'image RGB aléatoire
    img_displayed = ax_img.imshow(rgb_image)


    # Fonction pour gérer l'événement de survol de la souris
    def on_hover(event):
        if event.inaxes == ax:
            contains, ind = scatter.contains(event)
            if contains:
                # Afficher l'image RGB aléatoire sur la deuxième figure
                ind = ind['ind'][0]  # Indice du point survolé
                ax_img.clear()  # Effacer l'image précédente
                ax_img.imshow(rgb_image)  # Afficher la nouvelle image
                plt.draw()
            else:
                # Effacer l'image lorsque la souris ne survole pas un point
                ax_img.clear()
                plt.draw()


    # Connecter la fonction à l'événement de survol de la souris
    cid = fig.canvas.mpl_connect('motion_notify_event', on_hover)

    plt.show()
