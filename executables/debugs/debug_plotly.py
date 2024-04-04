import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image

def generate_random_image():
    # Générer une image aléatoire
    random_image = np.random.randint(0, 256, size=(200, 200, 3), dtype=np.uint8)
    return random_image

def plot_scatter():
    # Générer des données pour le nuage de points
    x = np.random.rand(100)
    y = np.random.rand(100)

    # Créer la figure pour le nuage de points
    scatter_fig = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        hoverinfo='skip',  # Désactiver l'affichage de l'info-bulle par défaut
        customdata=np.arange(len(x)),  # Données personnalisées pour chaque point
    )

    return scatter_fig

def plot_image(random_image):
    # Convertir l'image numpy en format PIL
    pil_image = Image.fromarray(random_image)

    # Enregistrer l'image au format temporaire
    pil_image_path = "random_image.png"
    pil_image.save(pil_image_path)

    # Créer la trace d'image pour Plotly
    image_fig = go.Image(z=pil_image)

    return image_fig

def update_image(trace, points, selector):
    # Régénérer une nouvelle image aléatoire
    random_image = generate_random_image()

    # Mettre à jour la trace de l'image
    fig.update_traces(z=random_image, selector=dict(type='image'))

if __name__ == "__main__":
    # Créer une fenêtre avec deux figures différentes
    fig = make_subplots(rows=1, cols=2)

    # Ajouter la première figure (nuage de points)
    scatter_fig = plot_scatter()
    fig.add_trace(scatter_fig, row=1, col=1)

    # Générer une image aléatoire initiale
    initial_image = generate_random_image()

    # Ajouter la deuxième figure (image)
    image_fig = plot_image(initial_image)
    fig.add_trace(image_fig, row=1, col=2)

    # Définir l'événement de survol du curseur pour le nuage de points
    scatter_fig.on_hover(update_image)

    # Afficher la fenêtre
    fig.show()
