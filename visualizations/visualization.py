import os

import cv2
from bokeh.layouts import grid
from bokeh.io import curdoc
import numpy as np
from sklearn.manifold import TSNE

from utilities.global_include import delete_directory, create_directory
from visualizations.combining_render_cluster import CombiningRenderCluster
from visualizations.panel_control import PanelControl
from visualizations.projection_latent_space import ProjectionLatentSpace
from visualizations.rendering_observation import RenderingObservation


def finding_number_clusters(classifier) -> int:
    number_cluster = None
    if hasattr(classifier, 'cluster_centers_'):
        number_cluster = classifier.cluster_centers_.shape[0] + 1
    return number_cluster


class Visualization:
    def __init__(self, visualization_directory):
        self.visualization_directory = visualization_directory
        delete_directory(self.visualization_directory)
        create_directory(self.visualization_directory)

        self.name_selected_classifier = None
        self.names_selected_cluster = []
        self.classifier_names = []
        self.projector_names = []

        self.classifiers = {}
        self.number_clusters_classifier = {}
        self.clusters_representation = {}
        self.data = None

        self.max_number_cluster = None
        self.number_cluster = None

        self.rendering_observation = RenderingObservation(self)
        self.projection_latent_space = ProjectionLatentSpace(self)
        self.panel_control = PanelControl(self)
        self.combining_render_cluster = CombiningRenderCluster(self)

    def set_data(self, data):
        self.data = data

    def set_classifier(self, classifiers):
        self.classifiers = classifiers

        self.max_number_cluster = 0
        for name in self.classifiers.keys():
            self.classifier_names.append(name)

            cluster_number = finding_number_clusters(self.classifiers[name])
            self.number_clusters_classifier[name] = cluster_number
            if self.max_number_cluster < cluster_number:
                self.max_number_cluster = cluster_number

    def classification(self):
        for name in self.classifier_names:
            classifier = self.classifiers[name]
            prediction = None
            if hasattr(classifier, 'predict'):
                prediction = classifier.predict(self.data['latent_space'])
            elif hasattr(classifier, 'fit_predict'):
                prediction = classifier.fit_predict(self.data['latent_space'])
            else:
                print('warning')

            self.data[name] = np.array(prediction)

    def generation_renderings(self):
        path = self.visualization_directory + '/rendering'
        delete_directory(path)
        create_directory(path)

        images = self.data['rendering']
        paths = []

        for i, img_array in enumerate(images):
            filename = f"rendering_{i}.png"
            filepath = os.path.join(path, filename)
            cv2.imwrite(filepath, np.array(img_array))
            delimiter = "Whitening_Agent_Black_Box"
            suffix = filepath.split(delimiter)[-1]
            path_with_delimiter = delimiter + suffix
            paths.append(path_with_delimiter)

        self.data['paths_rendering'] = paths
        self.projection_latent_space.source.data['paths_rendering'] = self.data['paths_rendering']

    def generate_clusters_representation(self):
        path = self.visualization_directory + '/clusters_representation'
        for classifier_name in self.classifier_names:
            paths = []
            for i in range(self.number_clusters_classifier[classifier_name]):
                indices = np.where(self.data[classifier_name] == i)
                cluster_rendering = self.data['rendering'][indices]
                cluster_rendering = np.sum(cluster_rendering, axis=0)
                cluster_rendering = np.clip(cluster_rendering, 0, 255)

                filename = f"cluster_{i}.png"
                create_directory(os.path.join(path, classifier_name))
                filepath = os.path.join(path, classifier_name, filename)
                cv2.imwrite(filepath, np.array(cluster_rendering))
                delimiter = "Whitening_Agent_Black_Box"
                suffix = filepath.split(delimiter)[-1]
                path_with_delimiter = delimiter + suffix
                paths.append(path_with_delimiter)

            self.clusters_representation[classifier_name] = paths

    def projection(self):
        tsne = TSNE(n_components=2)
        self.data['tsne_projection'] = tsne.fit_transform(self.data['latent_space'])
        self.projector_names.append('tsne_projection')

    def selection_projector(self, projector_name):
        self.projection_latent_space.source.data['x'] = self.data[projector_name][:, 0]
        self.projection_latent_space.source.data['y'] = self.data[projector_name][:, 1]

    def selection_classifier(self, classifier_name):
        self.name_selected_classifier = classifier_name
        self.number_cluster = finding_number_clusters(self.classifiers[classifier_name])
        self.selection_cluster([])
        self.projection_latent_space.source.data['cluster'] = self.data[classifier_name].astype(str)
        self.panel_control.update()

    def selection_cluster(self, cluster_names):
        self.names_selected_cluster = cluster_names
        self.panel_control.update()
        if len(self.names_selected_cluster) > 0:
            cluster_indices = []
            for index, label_cluster in enumerate(self.projection_latent_space.source.data['cluster']):
                if label_cluster in self.names_selected_cluster:
                    cluster_indices.append(index)

            self.projection_latent_space.source.selected.indices = cluster_indices
            # for cluster_name in self.names_selected_cluster:
            self.panel_control.slider_selection_cluster_render.value = np.clip(self.panel_control.slider_selection_cluster_render.value, 0, len(self.names_selected_cluster)-1)
            self.selection_render_cluster(self.panel_control.slider_selection_cluster_render.value)
        else:
            self.projection_latent_space.source.selected.indices = []
            self.combining_render_cluster.source.data['rendering_path'] = []

    def selection_render_cluster(self, number):
        self.combining_render_cluster.source.data['rendering_path'] = [self.clusters_representation[self.name_selected_classifier][int(self.names_selected_cluster[number])]]

    def initialisation(self):
        self.projection()
        self.classification()
        self.selection_projector(self.projector_names[0])
        self.selection_classifier(self.classifier_names[0])
        self.projection_latent_space.update()
        self.panel_control.update()

    def display(self):
        curdoc().add_root(grid([[self.projection_latent_space.figure, self.rendering_observation.figure], [self.panel_control.layer, self.combining_render_cluster.figure]]))
        curdoc().title = 'Visualisations'