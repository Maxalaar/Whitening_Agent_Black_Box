from bokeh.transform import factor_cmap
from bokeh.models import CustomJS, HoverTool, ResetTool, WheelZoomTool, PanTool, ColumnDataSource
from bokeh.plotting import figure
import numpy as np
import colorcet


class ProjectionLatentSpace:
    def __init__(self, visualization):
        self.name = 'Projection of the latent space'
        self.visualization = visualization
        self.source = ColumnDataSource({})
        self.color_map = None

        code = """
            const indices = cb_data.index.indices;
            console.log(indices);
            for (let i = 0; i < indices.length; i++) {
                const index = indices[i];
                render_source.data['path_rendering'] = [source.data['paths_rendering'][index]];
                render_source.change.emit();
            }
        """
        callback = CustomJS(
            args={'source': self.source, 'render_source': self.visualization.rendering_observation.source},
            code=code
        )

        hover = HoverTool(
            tooltips=[('Cluster', '@cluster')],
            callback=callback,
        )

        wheel_zoom = WheelZoomTool()
        pan = PanTool()
        reset = ResetTool()
        tools = (wheel_zoom, pan, hover, reset)
        self.figure = figure(tools=tools, toolbar_location='below', title=self.name)
        self.figure.toolbar.active_scroll = wheel_zoom

    def update(self):
        self.color_map = factor_cmap(
            field_name='cluster',
            palette=colorcet.glasbey[:self.visualization.max_number_cluster],
            factors=np.arange(self.visualization.max_number_cluster).astype(str),
        )
        self.figure.circle(source=self.source, x='x', y='y', radius=4, radius_units='screen', fill_color=self.color_map, line_color=None)
