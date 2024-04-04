from bokeh.plotting import figure, curdoc
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, HoverTool, CustomJS

# Créer une source de données pour l'image
image_source = ColumnDataSource(data=dict(image_url=["https://docs.bokeh.org/static/snake.jpg",
                                                      "https://docs.bokeh.org/static/snake.jpg",
                                                      "https://docs.bokeh.org/static/snake.jpg",
                                                      "https://docs.bokeh.org/static/snake4_TheRevenge.png"]))

# Créer une source de données pour le nuage de points
points_source = ColumnDataSource(data=dict(x=[0.1, 0.3, 0.5], y=[0.1, 0.5, 0.3]))

# Créer la figure contenant l'image
image_figure = figure(x_range=(0, 1), y_range=(0, 1), width=400, height=400)
image_figure.image_url(url='image_url', x=0, y=1, w=1, h=1, source=image_source)

# Créer une nouvelle figure pour le nuage de points
scatter_figure = figure(x_range=(0, 1), y_range=(0, 1), width=400, height=400, tools="hover")
scatter_renderer = scatter_figure.circle(x='x', y='y', source=points_source, size=10, color='red')

# Créer une fonction de callback pour mettre à jour l'image lorsqu'un point est survolé
hover_callback = CustomJS(args=dict(image_source=image_source), code="""
    const indices = cb_data.index.indices;
    for (let i = 0; i < indices.length; i++) {
        image_source.data['image_url'] = image_source.data['image_url'][indices[i]];
        image_source.change.emit();
    }
""")

# Associer la fonction de callback au survol des points dans la figure du nuage de points
hover_tool = HoverTool(renderers=[scatter_renderer], callback=hover_callback)
scatter_figure.add_tools(hover_tool)

# Créer une disposition des figures
layout = row(image_figure, scatter_figure)

# Ajouter la disposition au document Bokeh
curdoc().add_root(layout)
