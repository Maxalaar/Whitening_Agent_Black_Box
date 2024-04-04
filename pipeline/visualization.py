import io
import os

import cv2
import numpy as np
import pandas as pd
from bokeh.io import output_file
from bokeh.layouts import column, row
from sklearn.manifold import TSNE

from bokeh.plotting import figure, show
from bokeh.transform import factor_cmap
from bokeh.models import ColumnDataSource, WheelZoomTool, PanTool, HoverTool, ResetTool, Div, CustomJS
import colorcet as cc

from utilities.dataset_handler import DatasetHandler
from utilities.global_include import delete_directory, create_directory
from utilities.sklearn_classifier_handler import SklearnClassifierHandler


class Visualisation:
    def __init__(self, visualization_directory):
        self.visualization_directory = visualization_directory
        output_file(visualization_directory + '/visualization.html')

        self.latent_space_data = None
        self.latent_space_rendering = None
        self.latent_space_labeled_by_cluster = None
        self.cluster_color_map = None

        self.global_source = None
        self.rendering_image_source = None

        self.latent_space_2d = None
        self.classifier = None
        self.number_cluster = None

        self.figure_latent_space_2d = None
        self.figure_render_2d = None

    def set_latent_space_data(self, latent_space_data):
        self.latent_space_data = latent_space_data

    def set_latent_space_rendering(self, set_latent_space_rendering):
        self.latent_space_rendering = set_latent_space_rendering

    def set_classifier(self, classifier):
        self.classifier = classifier

    def projection_latent_space_2d(self):
        tsne = TSNE(n_components=2)
        self.latent_space_2d = tsne.fit_transform(self.latent_space_data)

    def classifier_predict(self):
        if hasattr(self.classifier, 'predict'):
            self.latent_space_labeled_by_cluster = self.classifier.predict(self.latent_space_data)
        elif hasattr(self.classifier, 'fit_predict'):
            self.latent_space_labeled_by_cluster = self.classifier.fit_predict(self.latent_space_data)
        else:
            print('warning')

        if hasattr(self.classifier, 'cluster_centers_'):
            self.number_cluster = self.classifier.cluster_centers_.shape[0] + 1

    def convert_bokeh_format(self):
        df = pd.DataFrame({
            'x_latent_space_2d': self.latent_space_2d[:, 0],
            'y_latent_space_2d': self.latent_space_2d[:, 1],
            'cluster': self.latent_space_labeled_by_cluster.astype(str),
            'rendering': [image for image in self.latent_space_rendering],
        })

        def str2int(cluster):
            return cluster.astype(int)

        df = df.sort_values(by='cluster', key=str2int)

        self.global_source = ColumnDataSource(df)

    def create_figure_latent_space_2d(self):
        path = self.visualization_directory + '/rendering'
        delete_directory(path)
        create_directory(path)
        images = self.global_source.data['rendering'].tolist()
        path_images = []

        for i, img_array in enumerate(images):
            filename = f"rendering_{i}.png"
            filepath = os.path.join(path, filename)
            cv2.imwrite(filepath, img_array)
            path_images.append(filepath)

        self.global_source.add(path_images, 'rendering_path')

        self.rendering_image_source = ColumnDataSource(data=dict(rendering_path=[]))

        self.figure_render_2d = figure(x_range=(0, 1), y_range=(0, 1), toolbar_location=None)
        self.figure_render_2d.axis.visible = False
        self.figure_render_2d.grid.visible = False

        self.figure_render_2d.image_url(url='rendering_path', x=0, y=1, w=1, h=1, source=self.rendering_image_source)

        wheel_zoom = WheelZoomTool()
        pan = PanTool()

        code = """
            const indices = cb_data.index.indices;
            for (let i = 0; i < indices.length; i++) {
                const index = indices[i];
                const rendering_path = global_source.data['rendering_path'][index];
                console.log(rendering_path);
                image_source.data['rendering_path'] = [global_source.data['rendering_path'][index]];
                image_source.change.emit();
                
                // window.open(rendering_path, '_blank');
                // console.log(typeof figure_render_2d);
                // console.log(Object.keys(figure_render_2d));
                // console.log(Object.getOwnPropertyNames(figure_render_2d));
                // figure_render_2d.image_url();
                // newWindow = window.open('https://docs.bokeh.org/static/snake.jpg', '_blank');
            }
        """

        callback = CustomJS(args={'figure_render_2d': self.figure_render_2d, 'global_source': self.global_source, 'image_source': self.rendering_image_source}, code=code)

        hover = HoverTool(
            tooltips=[('Cluster', '@cluster')],
            callback=callback,
        )

        reset = ResetTool()
        tools = (wheel_zoom, pan, hover, reset)
        # tools = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,examine,help"

        self.figure_latent_space_2d = figure(tools=tools, toolbar_location="below", title='Representation of the latent space in 2D projection')
        self.figure_latent_space_2d.toolbar.active_scroll = wheel_zoom

        self.cluster_color_map = factor_cmap('cluster', palette=cc.glasbey[:self.number_cluster], factors=np.arange(self.number_cluster).astype(str))
        self.figure_latent_space_2d.circle(source=self.global_source, x='x_latent_space_2d', y='y_latent_space_2d', radius=5, radius_units='screen', fill_color=self.cluster_color_map, line_color=None, legend_field='cluster')
        self.figure_latent_space_2d.legend.title = 'Cluster'
        self.figure_latent_space_2d.legend.location = 'top_right'

    def show(self):
        layout = row(self.figure_latent_space_2d, self.figure_render_2d)
        show(layout)


def visualization(visualization_directory, datasets_directory, sklearn_directory):
    print('-- Visualization --')
    print()
    delete_directory(visualization_directory)
    create_directory(visualization_directory)

    data = DatasetHandler(datasets_directory, 'latent_space').load(['latent_space', 'action', 'rendering'], number_data=2000)
    latent_space = data['latent_space']
    rendering = data['rendering']

    visualization = Visualisation(visualization_directory)
    visualization.set_latent_space_data(latent_space)
    visualization.set_latent_space_rendering(rendering)
    visualization.set_classifier(SklearnClassifierHandler(sklearn_directory, 'mean_shift_classifier').load())

    visualization.projection_latent_space_2d()
    visualization.classifier_predict()

    visualization.convert_bokeh_format()
    visualization.create_figure_latent_space_2d()

    visualization.show()
