from bokeh.models import ColumnDataSource
from bokeh.plotting import figure


class RenderingObservation:
    def __init__(self, visualization):
        self.name = 'Rendering of the Observation'
        self.visualization = visualization
        self.source = ColumnDataSource({})
        self.figure = figure(tools='', x_range=(0, 1), y_range=(0, 1), toolbar_location=None, title=self.name)
        self.figure.axis.visible = False
        self.figure.grid.visible = False
        self.figure.image_url(url='path_rendering', x=0, y=1, w=1, h=1, source=self.source)
        # self.figure.image_url(url=['https://docs.bokeh.org/static/snake.jpg'], x=0, y=1, w=1, h=1)