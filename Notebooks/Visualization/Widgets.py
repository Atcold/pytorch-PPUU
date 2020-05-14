"""Widgets that make up the visualizations

All these widgets are not supposed to be used alone, they are joined
to form visualizations, for example episode picker and episode review.

All widgets can roughly be split into two types: pickers and viewers.

Pickers are widgets that allow us to pick what to visualize, such as
Picker widget, or DimensionalityReduction widget.

Viewers are widgets that are built for viewing data for the selected
experiment, model, or episode. Examples include success plot, episode
viewer.
"""

import ipywidgets as widgets
import numpy as np
from bqplot.marks import Pie, GridHeatMap, Lines, Scatter
from bqplot import Figure
from bqplot.scales import LinearScale, ColorScale, OrdinalScale
import bqplot as bq

from collections import OrderedDict
import traitlets

from DataReader import DataReader
from DimensionalityReduction import DimensionalityReduction


class Picker(widgets.VBox):
    """Picker widget
    This widget is in essence a set of drop-down menus.
    They allow, in turns, to choose experiment, seed, checkpoint, and episode.

    The picker can be created for different levels of granularity:
        EXPERIMENT_LEVEL is only one dropdown that lets us choose the
                         experiment.
        MODEL_LEVEL includes dropdowns for experiment, seed, and checkpoint and
                    lets us choose the model
        EPISODE_LEVEL includes dropdowns for experiment, seed, checkpoint,
                      and episode, letting us choose evaluation of a model
                      on a given episode.
    """
    EXPERIMENT_LEVEL = 0
    MODEL_LEVEL = 1
    EPISODE_LEVEL = 2

    def __init__(self, level, callback=None, widget=None):
        """
        Args:
            level: int, EXPERIMENT_LEVEL, MODEL_LEVEL, or EPISODE_LEVEL.
                   speicifes what this picker should pick.
            callback: if not none, when the user picks all the values
                      the callback will be called.
                      Note that for different levels the number of
                      parameters passed to callback is different.
            widget: if not none, when the user picks all the values
                    widget.update(*args) will be called.
                    Note that for different levels the number of
                    parameters passed to update is different.
        """
        children = []
        self.experiment_dropdown = widgets.Dropdown(
            options=list(DataReader.get_experiments_mapping().keys()),
            description='Experiment:',
            disabled=False,
            value=None,
        )
        self.callback = callback
        self.widget = widget

        children.append(self.experiment_dropdown)

        if level >= Picker.MODEL_LEVEL:

            self.seed_dropdown = widgets.Dropdown(
                description='Seed:',
                disabled=True,
            )
            children.append(self.seed_dropdown)

            self.checkpoint_dropdown = widgets.Dropdown(
                description='Checkpoint:',
                disabled=True,
            )
            children.append(self.checkpoint_dropdown)

        if level >= Picker.EPISODE_LEVEL:
            self.episode_dropdown = widgets.Dropdown(
                description='Episode:',
                disabled=True,
            )
            children.append(self.episode_dropdown)

        self.ignore_updates = False

        def experiment_dropdown_change_callback(change):
            if change.name == 'value' and change.new is not None:
                self.ignore_updates = True
                if level >= Picker.MODEL_LEVEL:
                    self.seed_dropdown.options = \
                        DataReader.find_option_values(
                            option='seed',
                            experiment=self.experiment_dropdown.value
                        )
                    self.seed_dropdown.value = None
                    self.seed_dropdown.disabled = False
                    self.checkpoint_dropdown.disabled = True
                    self.checkpoint_dropdown.value = None
                self.ignore_updates = False
                if level == Picker.EXPERIMENT_LEVEL:
                    self.call_callback(self.experiment_dropdown.value)

        def seed_dropdown_change_callback(change):
            if self.ignore_updates:
                return
            if change.name == 'value' and change.new is not None:
                self.ignore_updates = True
                self.checkpoint_dropdown.options = \
                    DataReader.find_option_values(
                        option='checkpoint',
                        experiment=self.experiment_dropdown.value,
                        seed=self.seed_dropdown.value
                    )
                self.checkpoint_dropdown.value = None
                self.checkpoint_dropdown.disabled = False
                self.ignore_updates = False

        def checkpoint_dropdown_change_callback(change):
            if self.ignore_updates:
                return

            if change.name == 'value' and change.new is not None:
                self.ignore_updates = True
                if level >= Picker.EPISODE_LEVEL:
                    self.episode_dropdown.options = \
                        DataReader.find_option_values(
                            option='episode',
                            experiment=self.experiment_dropdown.value,
                            seed=self.seed_dropdown.value,
                            checkpoint=self.checkpoint_dropdown.value
                        )
                    self.episode_dropdown.value = None
                    self.episode_dropdown.disabled = False
                self.ignore_updates = False

                if level == Picker.MODEL_LEVEL:
                    self.call_callback(self.experiment_dropdown.value,
                                       self.seed_dropdown.value,
                                       self.checkpoint_dropdown.value)

        def episode_dropdown_change_callback(change):
            if self.ignore_updates:
                return
            if change.name == 'value' and change.new is not None:
                if level == Picker.EPISODE_LEVEL:
                    self.call_callback(self.experiment_dropdown.value,
                                       self.seed_dropdown.value,
                                       self.checkpoint_dropdown.value,
                                       self.episode_dropdown.value)

                    self.experiment_dropdown.observe(
                        experiment_dropdown_change_callback,
                        type='change'
                    )

        self.experiment_dropdown.observe(experiment_dropdown_change_callback,
                                         type='change')

        if level >= Picker.MODEL_LEVEL:
            self.seed_dropdown.observe(seed_dropdown_change_callback,
                                       type='change')
            self.checkpoint_dropdown.observe(
                checkpoint_dropdown_change_callback,
                type='change'
            )

        if level >= Picker.EPISODE_LEVEL:
            self.episode_dropdown.observe(episode_dropdown_change_callback,
                                          type='change')

        super(Picker, self).__init__(children)

    def call_callback(self, *args):
        if self.callback is not None:
            self.callback(*args)
        if self.widget is not None:
            self.widget.update(*args)

    def get_selected_experiment(self):
        return self.experiment_dropdown.value

    def get_selected_seed(self):
        return self.seed_dropdown.value

    def get_selected_checkpoint(self):
        return self.checkpoint_dropdown.value

    def get_selected_episode(self):
        return self.episode_dropdown.value


class PolicyComparison(widgets.VBox):
    """Widget for comparing success rates across checkpoints of different
    policies.

    Contains a single plot of success rates.
    """

    def __init__(self):
        # self.experiment_plot = plt.subplots(figsize=(18, 4))
        # self.experiment_plot_output = widgets.Output()
        self.x_sc = LinearScale()
        self.y_sc = LinearScale()
        ax_x = bq.Axis(label='steps',
                       scale=self.x_sc,
                       grid_lines='solid')
        ax_y = bq.Axis(label='success rate',
                       scale=self.y_sc,
                       orientation='vertical',
                       side='left',
                       grid_lines='solid')
        self.scales = {'x': self.x_sc, 'y': self.y_sc}
        self.experiment_figure = Figure(
            title='comparison',
            marks=[],
            layout=widgets.Layout(width='100%'),
            axes=[ax_x, ax_y],
            legend_location='top-right',
        )
        super(PolicyComparison, self).__init__([self.experiment_figure])

    def update(self, experiments):
        """updates the plots for given experiments

        Args:
            experiments: array of strings. Names of experiments to be loaded.
        """
        marks = []
        colors = bq.colorschemes.CATEGORY10
        for i, experiment in enumerate(experiments):
            steps, result = DataReader.get_success_rates_for_experiment(
                experiment)
            x = np.array(steps)
            c = colors[i]
            between_fill = Lines(x=[x, x],
                                 y=[result.min(0), result.max(0)],
                                 fill='between',
                                 colors=[c, c],
                                 opacities=[0.1, 0.1],
                                 fill_colors=[c],
                                 fill_opacities=[0.3],
                                 scales=self.scales,
                                 )
            line = Lines(x=x, y=np.median(result, 0),
                         scales=self.scales,
                         colors=[c],
                         display_legend=True,
                         labels=[experiment],
                         )
            marks.append(between_fill)
            marks.append(line)

        self.experiment_figure.marks = marks
        return

class LearningCurve(widgets.VBox):
    """Widget for comparing learning curves of different policies.

    Contains a single plot of learning curves across timesteps.
    """

    def __init__(self):
        # self.experiment_plot = plt.subplots(figsize=(18, 4))
        # self.experiment_plot_output = widgets.Output()
        self.x_sc = LinearScale()
        self.y_sc = LinearScale()
        ax_x = bq.Axis(label='steps',
                       scale=self.x_sc,
                       grid_lines='solid')
        ax_y = bq.Axis(label='loss',
                       scale=self.y_sc,
                       orientation='vertical',
                       side='left',
                       grid_lines='solid')
        self.scales = {'x': self.x_sc, 'y': self.y_sc}
        self.experiment_figure = Figure(
            title='comparison',
            marks=[],
            layout=widgets.Layout(width='100%'),
            axes=[ax_x, ax_y],
            legend_location='top-right',
        )
        super(LearningCurve, self).__init__([self.experiment_figure])

    def update(self, experiments):
        """updates the plots for given experiments

        Args:
            experiments: array of strings. Names of experiments to be loaded.
        """
        marks = []
        colors = bq.colorschemes.CATEGORY10
        for i, experiment in enumerate(experiments):
            curves = DataReader.get_learning_curves_for_experiment(experiment)
            x = np.array(curves['steps'])
            def draw_line(result, std, label, c):
                between_fill = Lines(x=[x, x],
                                     y=[result - std, result + std],
                                     fill='between',
                                     colors=[c, c],
                                     opacities=[0.1, 0.1],
                                     fill_colors=[c],
                                     fill_opacities=[0.3],
                                     scales=self.scales,
                                     )
                self.y_sc.max = np.min([np.max(result), np.median(result) * 2])
                self.y_sc.min = max(np.min(result), np.mean(result) - 2 * np.std(result))
                line = Lines(x=x, y=result,
                             scales=self.scales,
                             colors=[c],
                             display_legend=True,
                             labels=[label],
                             )
                marks.append(between_fill)
                marks.append(line)
            draw_line(np.array(curves['train'][0], dtype=np.float),
                      np.array(curves['train'][1], dtype=np.float),
                      experiment + '_train',
                      colors[2 * i])
            draw_line(np.array(curves['validation'][0], dtype=np.float),
                      np.array(curves['validation'][1], dtype=np.float),
                      experiment + '_validation',
                      colors[2 * i + 1])

        self.experiment_figure.marks = marks
        return


class EpisodeReview(widgets.VBox):
    """Widget which allows to view the episode in detail.
    Contains video visualization for simulator images, gradient images,
    and a plot that shows current costs and speed.
    """

    def __init__(self):
        self.episode_play = widgets.Play(
            value=0,
            min=0,
            max=100,
            step=1,
            description="Press play",
            disabled=False,
            interval=10,
        )
        self.update_interval_slider = widgets.IntSlider(
            min=1,
            max=300,
            value=30,
        )
        self.update_interval_box = widgets.HBox([
            widgets.Label("Animation update interval:"),
            self.update_interval_slider
        ])
        self.episode_slider = widgets.IntSlider()
        self.episode_hbox = widgets.HBox([
            self.episode_play,
            self.episode_slider
        ])
        self.episode_vbox = widgets.VBox([
            self.episode_hbox,
            self.update_interval_box
        ])

        widgets.jslink((self.episode_play, 'value'),
                       (self.episode_slider, 'value'))
        widgets.jslink((self.episode_play, 'max'),
                       (self.episode_slider, 'max'))
        widgets.jslink((self.episode_play, 'min'),
                       (self.episode_slider, 'min'))

        widgets.jslink((self.update_interval_slider, 'value'),
                       (self.episode_play, 'interval'))

        self.episode_gradient_image = widgets.Image(format='png',
                                                    width=120,
                                                    height=600,
                                                    )
        self.episode_image = widgets.Image(format='png',
                                           width=120,
                                           height=600,
                                           )

        x_sc = bq.LinearScale()
        # x_sc.max = size * 1.3
        y_sc = bq.LinearScale()
        y_sc.max = 1
        y_sc2 = bq.LinearScale()

        self.x_sc = x_sc

        ax_x = bq.Axis(label='step', scale=x_sc, grid_lines='none')
        ax_x.min = 0
        ax_x.max = 100
        ax_y = bq.Axis(label='costs',
                       scale=y_sc,
                       orientation='vertical',
                       grid_lines='solid')
        ax_y2 = bq.Axis(label='speed',
                        scale=y_sc2,
                        orientation='vertical',
                        side='right',
                        grid_lines='none')

        self.costs_plot_lines_costs = Lines(
            scales={'x': x_sc, 'y': y_sc}, display_legend=True, stroke_width=1)
        self.costs_plot_lines_speed = Lines(scales={'x': x_sc, 'y': y_sc2},
                                            colors=['red'],
                                            display_legend=True,
                                            stroke_width=1)
        self.costs_plot_progress = Lines(scales={'x': x_sc, 'y': y_sc})

        pan_zoom = bq.interacts.PanZoom(scales={'x': [x_sc], 'y': []})

        self.costs_plot_figure = Figure(marks=[self.costs_plot_lines_costs,
                                               self.costs_plot_lines_speed,
                                               self.costs_plot_progress],
                                        axes=[ax_x, ax_y, ax_y2],
                                        title='Costs and speed',
                                        legend_location='top-left',
                                        interaction=pan_zoom,
                                        layout=widgets.Layout(width='100%', height='100%')
                                       )

        self.follow_present = widgets.ToggleButton(
            value=False,
            description='Follow present',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        )

        self.reset_scale = widgets.Button(
            description='Reset scale',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        )

        def reset_scale_callback(b):
            self.reset_costs_plot_scale()

        self.reset_scale.on_click(reset_scale_callback)

        self.plot_controls_hbox = widgets.HBox(
            [self.follow_present, self.reset_scale],
            layout=widgets.Layout(width='100%', justify_content='center')
        )

        self.costs_plot_box = widgets.VBox(
            [self.costs_plot_figure, self.plot_controls_hbox],
            layout=widgets.Layout(width='100%')
        )

        self.images_hbox = widgets.HBox(
            [
                self.episode_gradient_image,
                self.episode_image,
                self.costs_plot_box
            ],
            layout=widgets.Layout(width='100%')
        )

        def episode_slider_callback(change):
            if change.name == 'value' and change.new is not None:
                gradient_shift = max(
                    0, len(self.images) - len(self.gradient_images))
                if change.new >= gradient_shift:
                    self.episode_gradient_image.value = self.gradient_images[
                        change.new - gradient_shift]
                self.episode_image.value = self.images[change.new]
                self.update_timestamp_line(change.new)

        self.episode_slider.observe(episode_slider_callback, type='change')

        super(EpisodeReview, self).__init__(
            [self.episode_vbox, self.images_hbox])

    def update_costs_plot(self, experiment, seed, checkpoint, episode):
        speeds = DataReader.get_episode_speeds(
            experiment, seed, checkpoint, episode)
        self.costs_plot_figure.title = f'Costs and speed: episode {episode}'
        if speeds is not None:
            self.costs_plot_lines_speed.x = range(len(speeds))
            self.costs_plot_lines_speed.y = speeds
            self.costs_plot_lines_speed.labels = ['speed']
        costs = DataReader.get_episode_costs(
            experiment, seed, checkpoint, episode)
        if costs is not None:
            x = costs.index
            self.x_sc.min = x[0]
            self.x_sc.max = x[-1]
            self.costs_plot_lines_costs.x = x
            self.costs_plot_lines_costs.y = [
                costs['lane_cost'],
                costs['pixel_proximity_cost'],
                costs['collisions_per_frame']
            ]
            self.costs_plot_lines_costs.labels = [
                'lane cost', 'pixel proximity cost', 'collisions per frame']

    def update_timestamp_line(self, timestamp):
        self.costs_plot_progress.x = [timestamp, timestamp]
        self.costs_plot_progress.y = [0, 1]
        if self.follow_present.value:
            self.x_sc.min = timestamp - 50
            self.x_sc.max = timestamp + 50

    def reset_costs_plot_scale(self):
        self.x_sc.min = None
        self.x_sc.max = None

    def update(self, experiment, seed, checkpoint, episode):
        """updates the plots and loads the images and the gradients, if
        they are available.

        This method is called by the corresponding picker.
        """
        self.gradient_images = DataReader.get_gradients(
            experiment, seed, checkpoint, episode)
        self.images = DataReader.get_images(
            experiment, seed, checkpoint, episode)

        if len(self.gradient_images) > 0:
            self.episode_gradient_image.value = self.gradient_images[0]
        if len(self.images) > 0:
            self.episode_image.value = self.images[0]

        self.episode_slider.min = 0
        self.episode_slider.value = 0
        self.episode_slider.max = len(self.images) - 1

        self.seed = seed
        self.checkpoint = checkpoint
        self.episode = episode

        self.reset_costs_plot_scale()
        self.update_costs_plot(experiment, seed, checkpoint, episode)


class DimensionalityReductionPlot(widgets.VBox):
    """Clustering picker for episodes for a given model.
    This contains a scatter plot for episodes, obtained by
    dimensionality reduction of different episode's evaluations.

    The goal is to help see main reasons of failures for a given
    model.
    """

    def __init__(self, callback=None, widget=None):
        self.callback = callback
        self.widget = widget

        self.DimensionalityReduction = DimensionalityReduction()
        self.x_scale = LinearScale()
        self.y_scale = LinearScale()

        pan_zoom = bq.interacts.PanZoom(
            scales={'x': [self.x_scale], 'y': [self.y_scale]})

        self.scatter = Scatter(
            scales={'x': self.x_scale, 'y': self.y_scale},
            default_opacities=[0.7],
            interactions={'click': 'select'},
            selected_style={
                'opacity': 1.0, 'stroke': 'Black'},
            unselected_style={'opacity': 0.5}
        )

        def scatter_callback(a, b):
            self.episode = b['data']['index'] + 1  # 1-indexed
            # we only have failures
            self.episode = self.failures_indices[self.episode - 1]
            if self.callback is not None:
                self.callback(self.experiment, self.seed,
                              self.step, self.episode)
            if self.widget is not None:
                self.widget.update(self.experiment, self.seed,
                                   self.step, self.episode)

        self.scatter.on_element_click(scatter_callback)
        self.toggle_buttons = widgets.ToggleButtons(
            options=OrderedDict([('Select', None), ('Zoom', pan_zoom)]))

        self.scatter_figure = Figure(
            marks=[self.scatter],
            layout=widgets.Layout(height='600px', width='100%')
        )

        traitlets.link((self.toggle_buttons, 'value'),
                       (self.scatter_figure, 'interaction'))

        super(DimensionalityReductionPlot, self).__init__(
            [self.scatter_figure, self.toggle_buttons])

    def update2(self, experiment, seed, step):
        # used in developement, not currently used
        self.experiment = experiment
        self.seed = seed
        self.step = step
        features = DimensionalityReduction.get_model_failing_features(
            experiment, seed, step)
        self.failures_indices = DataReader.get_episodes_with_outcome(
            experiment, seed, step, 0)

        failure_features = features[np.array(failures[:-1]) - 1]

        res = self.DimensionalityReduction.transform(features)
        colors = ['gray'] * res.shape[0]
        opacities = [0.3] * res.shape[0]

        classes = self.DimensionalityReduction.cluster(failure_features)

        category = bq.colorschemes.CATEGORY20[2:]
        for i, f in enumerate(failures):
            if f - 1 < len(colors):  # TODO: wtf?
                if i < len(classes):
                    colors[f - 1] = category[classes[i]]
                else:
                    colors[f - 1] = 'red'
                opacities[f - 1] = 0.8

        self.scatter.x = res[:, 0]
        self.scatter.y = res[:, 1]
        self.scatter.colors = colors
        self.scatter.opacity = opacities

    def update(self, experiment, seed, step):
        """updates the scatter plot.
        This method is called by the model picker """
        self.experiment = experiment
        self.seed = seed
        self.step = step
        self.failures_indices = DataReader.get_episodes_with_outcome(
            experiment, seed, step, 0)

        features = DimensionalityReduction.get_model_failing_features(
            experiment, seed, step)

        costs = DataReader.get_model_states(experiment, seed, step)
        print('costs shape', len(costs))
        print('max failure indices', max(self.failures_indices))

        res = self.DimensionalityReduction.transform(features)
        colors = ['gray'] * res.shape[0]
        opacities = [0.3] * res.shape[0]

        classes = self.DimensionalityReduction.cluster(features)

        category = bq.colorschemes.CATEGORY10[2:]
        for f in range(len(features)):
            if f - 1 < len(colors):  # TODO: wtf?
                if f < len(classes):
                    colors[f] = category[classes[f]]
                else:
                    colors[f] = 'red'
                opacities[f] = 0.8

        self.scatter.x = res[:, 0]
        self.scatter.y = res[:, 1]
        self.scatter.colors = colors
        self.scatter.opacity = opacities


class PiePlot(widgets.VBox):
    """A simple pie plot widget showing
    success vs failure rates for a model.
    """

    def __init__(self):
        self.pie_plot = Pie(radius=150,
                            inner_radius=80,
                            interactions={'click': 'select'},
                            colors=['green', 'red'],
                            label_color='black',
                            font_size='14px',
                            )
        self.pie_figure = Figure(title='Success rate', marks=[self.pie_plot])
        super(PiePlot, self).__init__([self.pie_figure])

    def update(self, experiment, seed, checkpoint):
        success_rate = DataReader.get_success_rate(
            experiment, seed, checkpoint)
        self.pie_plot.sizes = [success_rate, 1 - success_rate]
        self.pie_plot.labels = [
            str(round(success_rate, 2)), str(round(1 - success_rate, 2))]


class HeatMap(widgets.VBox):
    """HeatMap widget showing episodes 'difficulty' for a given model
    Each cell in the heatmap represents how hard an episode is"""

    def __init__(self):
        self.episode_grid_heat_map = GridHeatMap(
            color=np.random.rand(11, 51) * 0,
            scales={
                'row': OrdinalScale(),
                'column': OrdinalScale(),
                'color': ColorScale()
            },
            display_legend=False)
        self.episode_grid_heat_map_label = widgets.Label()
        self.episode_grid_heat_map_figure = Figure(
            title='Episode grid heat map',
            marks=[self.episode_grid_heat_map],
            layout=widgets.Layout(height='331px', width='100%')
        )

        def heat_map_click_callback(a, b):
            if self.result_permutation is not None:
                episode = self.result_permutation[b['data']['_cell_num']] + 1
                color = b['data']['color']
                self.episode_grid_heat_map_label.value = \
                    f'clicked on episode {episode} with {color}'\
                    'successful cases'

        self.episode_grid_heat_map.on_click(heat_map_click_callback)
        self.episode_grid_heat_map.on_element_click(heat_map_click_callback)

        super(HeatMap, self).__init__(
            [
                self.episode_grid_heat_map_figure,
                self.episode_grid_heat_map_label
            ]
        )

    def update(self, experiment):
        result = DataReader.get_episodes_success_counts(experiment)
        self.result_permutation = np.argsort(result)
        result = np.sort(result)
        self.episode_grid_heat_map.color = result.reshape(11, 51)
        self.color_result = result


class HeatMapComparison(widgets.VBox):
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
        self.episode_grid_heat_map = GridHeatMap(
            color=np.random.rand(11, 51) * 0,
            scales={
                'row': OrdinalScale(),
                'column': OrdinalScale(),
                'color': ColorScale(
                    colors=['orange', 'red', 'green', 'blue'],
                    min=0,
                    max=3,
                )
            },
            display_legend=False)
        self.episode_grid_heat_map_help = widgets.HTML(
            value='<br><br><ul>\
                    <li>orange - both models failed.</li>\
                    <li>red - first model succeeded, second failed</li>\
                    <li>green - first model failed, second succeeded</li>\
                    <li>blue - both models succeeded.</li>\
                    </ul>'
        )
        self.episode_grid_heat_map_label = widgets.Label()
        self.episode_grid_heat_map_figure = Figure(
            title='Comparison grid heat map',
            marks=[self.episode_grid_heat_map],
            layout=widgets.Layout(height='331px', width='100%')
        )

        def heat_map_click_callback(a, b):
            if self.result_permutation is not None:
                episode = self.result_permutation[b['data']['_cell_num']] + 1
                color = b['data']['color']
                self.episode_grid_heat_map_label.value = \
                    f'clicked on episode {episode} with {color}'\
                    'successful cases'

        self.episode_grid_heat_map.on_click(heat_map_click_callback)
        self.episode_grid_heat_map.on_element_click(heat_map_click_callback)

        super(HeatMapComparison, self).__init__(
            [
                self.episode_grid_heat_map_help,
                self.episode_grid_heat_map_figure,
                self.episode_grid_heat_map_label
            ]
        )

    def update(self, model1, model2):
        """
        Args:
            model1 - tuple containing (experiment, seed, step) for the first
                     compared model
            model2 - tuple containing (experiment, seed, step) for the second
                     compared model.
        """

        success_map_1 = DataReader.get_episode_success_map(*model1)
        success_map_2 = DataReader.get_episode_success_map(*model2)

        result = success_map_1 + 2 * success_map_2
        self.result_permutation = np.argsort(result)
        result = np.sort(result)
        self.episode_grid_heat_map.color = result.reshape(11, 51)
        self.color_result = result


class ExperimentEntryView(widgets.VBox):
    """A widget used to edit the directory of experiments for visualization.
    Contains edit text widgets for experiment name, root path and model prefix.
    """

    def __init__(self, name, experiment_root, model_name):
        style = {'description_width': 'initial'}
        self.experiment_name = widgets.Text(
            value=name,
            placeholder='name',
            description='Experiment name',
            style=style,
            layout=widgets.Layout(width='auto'),
        )
        self.experiment_root = widgets.Textarea(
            value=experiment_root,
            placeholder='path',
            description='Experiment root:',
            style=style,
            disabled=False,
            layout=widgets.Layout(width='auto'),
        )
        self.model_name = widgets.Textarea(
            value=model_name,
            placeholder='name',
            description='Model name prefix:',
            style=style,
            disabled=False,
            layout=widgets.Layout(width='auto'),
        )
        super(ExperimentEntryView, self).__init__(
            [
                self.experiment_name,
                self.experiment_root,
                self.model_name,
            ],
            layout=widgets.Layout(width='400px'),
        )
