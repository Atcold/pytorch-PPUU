"""Contains the tabs for different visualizations.
Each tab is a standalone visualization."""
from DataReader import DataReader
import ipywidgets as widgets
import json
import traitlets

from Widgets import (
    Picker,
    EpisodeReview,
    DimensionalityReductionPlot,
    PiePlot,
    HeatMap,
    HeatMapComparison,
    PolicyComparison,
    LearningCurve,
    ExperimentEntryView,
)


class EpisodeReviewTab(widgets.VBox):
    """A tab for visualizing model's performance on an episode.
    Model is picked with dropdown picker"""

    def __init__(self):
        self.episode_review = EpisodeReview()
        self.picker = Picker(
            Picker.EPISODE_LEVEL, widget=self.episode_review)
        super(EpisodeReviewTab, self).__init__(
            [self.picker, self.episode_review])


class PiePlotTab(widgets.VBox):
    """A tab for visualizing model's success rate with pie chart.
    Model is picked with dropdown picker"""

    def __init__(self):
        self.pie_plot = PiePlot()
        self.picker = Picker(Picker.MODEL_LEVEL,
                             widget=self.pie_plot)
        super(PiePlotTab, self).__init__([self.picker, self.pie_plot])


class DimensionalityReductionPlotTab(widgets.VBox):
    """A tab for visualizing episodes using with scatter plot and
    dimensionality reduction."""

    def __init__(self):
        self.episode_review = EpisodeReview()
        self.dimensionality_reduction_plot = DimensionalityReductionPlot(
            widget=self.episode_review)
        self.picker = Picker(Picker.MODEL_LEVEL,
                             widget=self.dimensionality_reduction_plot)
        super(DimensionalityReductionPlotTab, self).__init__(
            [self.picker, self.dimensionality_reduction_plot, self.episode_review])


class HeatMapTab(widgets.VBox):
    """A tabl showing episodes 'difficulty' for a given model
    Each cell in the heatmap represents how hard an episode is"""

    def __init__(self):
        self.heat_map = HeatMap()
        self.picker = Picker(
            Picker.EXPERIMENT_LEVEL, widget=self.heat_map)

        super(HeatMapTab, self).__init__([self.picker, self.heat_map])


class HeatMapComparisonTab(widgets.VBox):
    """Compares two models in their performance for different
    episodes, enabling us to see which episodes were failing or successful
    for two models.
    Color coding:
        orange - both models failed.
        red - first model succeeded, second failed.
        green - first model failed, second succeeded,
        blue - both models succeeded.
    """

    def __init__(self):

        self.heat_map = HeatMapComparison()
        self.picker0 = Picker(
            Picker.MODEL_LEVEL, callback=self.get_callback(0))
        self.picker1 = Picker(
            Picker.MODEL_LEVEL, callback=self.get_callback(1))

        self.picked_values = [None, None]

        self.pickers_hbox = widgets.HBox([self.picker0, self.picker1])

        super(HeatMapComparisonTab, self).__init__(
            [self.pickers_hbox, self.heat_map])

    def get_callback(self, index):
        def callback(episode, seed, step):
            self.picked_values[index] = (episode, seed, step)
            if self.picked_values[0] is not None and \
               self.picked_values[1] is not None:
                self.heat_map.update(
                    self.picked_values[0], self.picked_values[1])

        return callback


class PolicyComparisonTab(widgets.VBox):
    """Tab for comparing success rates across checkpoints of different
    experiments.

    Experiments are chosen using a multiselect widget.
    """

    def __init__(self):
        self.experiment_multiselect = widgets.SelectMultiple(
            options=list(DataReader.get_experiments_mapping().keys()),
            description='Experiments:',
            disabled=False,
            value=[],
        )

        self.policy_comparison = PolicyComparison()

        def experiment_multiselect_change_callback(change):
            if change.name == 'value' and change.new is not None:
                self.policy_comparison.update(change.new)

        self.experiment_multiselect.observe(
            experiment_multiselect_change_callback, type='change')
        super(PolicyComparisonTab, self).__init__(
            [self.experiment_multiselect, self.policy_comparison])


class LearningCurvesTab(widgets.VBox):
    """Tab for comparing learning curves for experiments.

    Experiments are chosen using a multiselect widget.
    """

    def __init__(self):
        self.experiment_multiselect = widgets.SelectMultiple(
            options=list(DataReader.get_experiments_mapping().keys()),
            description='Experiments:',
            disabled=False,
            value=[],
        )

        self.learning_curve = LearningCurve()

        def experiment_multiselect_change_callback(change):
            if change.name == 'value' and change.new is not None:
                self.learning_curve.update(change.new)

        self.experiment_multiselect.observe(
            experiment_multiselect_change_callback, type='change')
        super(LearningCurvesTab, self).__init__(
            [self.experiment_multiselect, self.learning_curve])


class ExperimentsDirectoryTab(widgets.HBox):
    """A tab that allows editing, deleting, and adding values to experiments
    directory.
    Contains a select widget, buttons and ExperimentEntryView.
    """

    def __init__(self):
        self.ignore_update = False

        def select_experiment_change_callback(change):
            if self.ignore_update:
                return
            if change.name == 'value' and change.new is not None:
                self.edit_experiment.children = [
                    self.widget_mapping[change.new]]

        def name_update_callback(_):
            self.update_selector()

        def save_callback(_):
            result_dict = {}
            for x in self.widget_mapping:
                root = self.widget_mapping[x].experiment_root.value
                model_name = self.widget_mapping[x].model_name.value
                result_dict[x] = [root, model_name]

            with open(DataReader.EXPERIMENTS_MAPPING_FILE, 'w') as f:
                json.dump(result_dict, f)

        def delete_callback(_):
            del self.widget_mapping[self.select_experiment.value]
            self.update_selector()

        def add_callback(_):
            self.widget_mapping['new'] = ExperimentEntryView('new', '', '')
            self.update_selector()
            self.select_experiment.value = 'new'

        with open(DataReader.EXPERIMENTS_MAPPING_FILE, 'r') as f:
            self.mapping = json.load(f)

        self.select_experiment = widgets.Select(
            options=self.mapping.keys(),
            disabled=False
        )
        self.save_button = widgets.Button(
            description='Save',
            disabled=False,
            layout=widgets.Layout(width='auto'),
        )
        self.delete_button = widgets.Button(
            description='Delete',
            disabled=False,
        )
        self.add_button = widgets.Button(
            description='Add',
            disabled=False,
        )
        self.buttons_hbox = widgets.HBox([self.add_button, self.delete_button])

        self.widget_mapping = {}
        for key in self.mapping:
            self.widget_mapping[key] = ExperimentEntryView(
                key,
                self.mapping[key][0],
                self.mapping[key][1]
            )
            self.widget_mapping[key].experiment_name.observe(
                name_update_callback)

        self.left_column = widgets.VBox(
            [
                self.select_experiment,
                self.buttons_hbox,
                self.save_button,
            ])
        self.edit_experiment = widgets.Box([])
        self.edit_experiment.layout.width = 'auto'

        self.select_experiment.observe(
            select_experiment_change_callback, type='change')

        self.add_button.on_click(add_callback)
        self.save_button.on_click(save_callback)
        self.delete_button.on_click(delete_callback)

        super(ExperimentsDirectoryTab, self).__init__(
            [self.left_column, self.edit_experiment],
            layout=widgets.Layout(width='100%', align_items='stretch'))

    def update_selector(self):
        """
        This functions serves to rebuild the selector with the
        actual values. Used when we add a new value, delete a value,
        and each time we change a the name
        of a given experiment. It updates the values in the selector, 
        preserving the selection.
        """
        self.ignore_update = True
        new_widget_mapping = {}
        options = []
        old_index = self.select_experiment.index
        for x in self.widget_mapping:
            name = self.widget_mapping[x].experiment_name.value
            new_widget_mapping[name] = self.widget_mapping[x]
            options.append(name)
        self.widget_mapping = new_widget_mapping
        self.select_experiment.options = options
        self.select_experiment.index = min(old_index, len(options) - 1)
        self.ignore_update = False
