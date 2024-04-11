from bokeh.models import Slider, Select, MultiChoice
from bokeh.layouts import column


class PanelControl:
    def __init__(self, visualization):
        self.visualization = visualization
        self.slider_selection_cluster_render = Slider(
            start=0,
            end=0,
            value=0,
            step=1,
            # title='Cluster',
        )
        self.multi_choice_selection_cluster = MultiChoice(title='Clusters selected :', options=[])
        self.checkbox_selection_classifier = Select(title='Classifier :', options=self.visualization.classifier_names)

        def callback_selection_cluster(attr, old, new):
            self.visualization.selection_cluster(new)
        self.multi_choice_selection_cluster.on_change('value', callback_selection_cluster)

        def callback_selection_classifier(attr, old, new):
            self.visualization.selection_classifier(new)
        self.checkbox_selection_classifier.on_change('value', callback_selection_classifier)

        def callback_selection_cluster_render(attr, old, new):
            self.visualization.selection_render_cluster(new)
        self.slider_selection_cluster_render.on_change('value', callback_selection_cluster_render)

        self.layer = column(self.checkbox_selection_classifier, self.multi_choice_selection_cluster, self.slider_selection_cluster_render)

    def update(self):
        self.checkbox_selection_classifier.options = self.visualization.classifier_names
        self.checkbox_selection_classifier.value = self.visualization.name_selected_classifier
        self.multi_choice_selection_cluster.options = [str(i) for i in range(self.visualization.number_cluster - 1)]

        if len(self.visualization.names_selected_cluster) > 0:
            self.multi_choice_selection_cluster.value = self.visualization.names_selected_cluster
            self.slider_selection_cluster_render.end = len(self.visualization.names_selected_cluster) - 1
        else:
            self.multi_choice_selection_cluster.value = []
            self.slider_selection_cluster_render.end = 0