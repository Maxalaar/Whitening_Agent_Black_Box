from bokeh.io import output_file
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.layouts import column
from bokeh.models.widgets import Slider

# Création des données pour la source de colonne
image_urls = ["/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/executables/debugs/snake.jpg", "/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/executables/debugs/snake.jpg"]
# "/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/executables/debugs/snake4_TheRevenge.png",
# "/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/executables/debugs/snake4_TheRevenge.png",
# "/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/executables/debugs/snake4_TheRevenge.png",
# "/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/executables/debugs/snake4_TheRevenge.png",
source = ColumnDataSource(data=dict(image_url=image_urls))

# Création de la figure
renderer_source = figure(x_range=(0, 1), y_range=(0, 1), width=400, height=400)
renderer_source.image_url(url='image_url', x=0, y=1, w=1, h=1, source=source)

# Callback JavaScript personnalisé pour changer l'image
callback = CustomJS(args=dict(source=source, renderer_source=renderer_source), code="""
    // new_source_data['x']
    // Récupérer les données de la source
    // var data = source.data;
    // Récupérer l'index de l'image actuellement affichée
    // var current_index = cb_obj.value;
    // Changer l'image en fonction de l'index
    // data['image_url'][3] = data['image_url'][current_index];
    // Mettre à jour la source
    renderer_source['image_url'] =  ["/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/results/pong_big_experiment/visualization/rendering/rendering_9.png", "/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/results/pong_big_experiment/visualization/rendering/rendering_9.png"]
    source.change.emit();
""")

# Création du widget pour sélectionner l'image
image_selection_widget = Slider(start=0, end=len(image_urls) - 1, step=1, value=0, title="Select Image")

# Associer le callback à l'événement de changement de valeur du widget
image_selection_widget.js_on_change('value', callback)

# Afficher la figure et le widget dans une disposition en colonne
layout = column(renderer_source, image_selection_widget)


def main():
    output_file(
        '/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/executables/debugs/debug_bokeh_v2.html')
    show(layout)


if __name__ == "__main__":
    main()
