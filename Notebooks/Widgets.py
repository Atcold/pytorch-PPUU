import ipywidgets as widgets
import numpy as np
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from bqplot.marks import Pie, GridHeatMap, Label, Lines, Bars, Scatter
from bqplot import Figure
from bqplot.scales import LinearScale, ColorScale, OrdinalScale
import bqplot as bq

from DataReader import DataReader
from PCA import PCA

class ModelPicker(widgets.VBox):
    EXPERIMENT_LEVEL = 0
    MODEL_LEVEL = 1
    EPISODE_LEVEL = 2

    def __init__(self, level, callback):
        children = []
        self.experiment_dropdown = widgets.Dropdown(
            options=list(DataReader.experiments_mapping.keys()),
            description='Experiment:',
            disabled=False,
            value=None,
        )

        children.append(self.experiment_dropdown)

        if level >= ModelPicker.MODEL_LEVEL:

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


        if level >= ModelPicker.EPISODE_LEVEL:
            self.episode_dropdown = widgets.Dropdown(
                description='Episode:',
                disabled=True,
            )
            children.append(self.episode_dropdown)

        self.ignore_updates = False

        def experiment_dropdown_change_callback(change):
            if change.name == 'value' and change.new is not None:
                self.ignore_updates = True
                if level >= ModelPicker.MODEL_LEVEL:
                    self.seed_dropdown.options = DataReader.find_option_values(option='seed', 
                                                                               experiment=self.experiment_dropdown.value)
                    self.seed_dropdown.value = None
                    self.seed_dropdown.disabled = False
                    self.checkpoint_dropdown.disabled = True
                    self.checkpoint_dropdown.value = None
                self.ignore_updates = False
                if level == ModelPicker.EXPERIMENT_LEVEL:
                    callback(self.experiment_dropdown.value)

        def seed_dropdown_change_callback(change):
            if self.ignore_updates:
                return
            if change.name == 'value' and change.new is not None:
                self.ignore_updates = True
                self.checkpoint_dropdown.options = DataReader.find_option_values(option='checkpoint', 
                                                                           experiment=self.experiment_dropdown.value,
                                                                           seed=self.seed_dropdown.value)
                self.checkpoint_dropdown.value = None
                self.checkpoint_dropdown.disabled = False
                self.ignore_updates = False

        def checkpoint_dropdown_change_callback(change):
            if self.ignore_updates:
                return

            if change.name == 'value' and change.new is not None:
                self.ignore_updates = True
                if level >= ModelPicker.EPISODE_LEVEL:
                    self.episode_dropdown.options = DataReader.find_option_values(option='episode', 
                                                                            experiment=self.experiment_dropdown.value,
                                                                            seed=self.seed_dropdown.value,
                                                                            checkpoint=self.checkpoint_dropdown.value)
                    self.episode_dropdown.value = None
                    self.episode_dropdown.disabled = False
                self.ignore_updates = False

                if level == ModelPicker.MODEL_LEVEL:
                    callback(self.experiment_dropdown.value, 
                             self.seed_dropdown.value,
                             self.checkpoint_dropdown.value)

        def episode_dropdown_change_callback(change):
            if self.ignore_updates:
                return
            if change.name == 'value' and change.new is not None:
                if level == ModelPicker.EPISODE_LEVEL:
                    callback(self.experiment_dropdown.value, 
                             self.seed_dropdown.value,
                             self.checkpoint_dropdown.value,
                             self.episode_dropdown.value)

        self.experiment_dropdown.observe(experiment_dropdown_change_callback, type='change')

        if level >= ModelPicker.MODEL_LEVEL:
            self.seed_dropdown.observe(seed_dropdown_change_callback, type='change')
            self.checkpoint_dropdown.observe(checkpoint_dropdown_change_callback, type='change')

        if level == ModelPicker.MODEL_LEVEL:
            callback('Deterministic policy, regressed cost', 3, 25000)

        if level >= ModelPicker.EPISODE_LEVEL:
            self.episode_dropdown.observe(episode_dropdown_change_callback, type='change')
            # TODO: remove
            callback('Deterministic policy, regressed cost', 3, 25000, 10)

        super(ModelPicker, self).__init__(children)

    def get_selected_experiment(self):
        return self.experiment_dropdown.value

    def get_selected_seed(self):
        return self.seed_dropdown.value

    def get_selected_checkpoint(self):
        return self.checkpoint_dropdown.value

    def get_selected_episode(self):
        return self.episode_dropdown.value


class PolicyComparison(widgets.VBox):

    def __init__(self):
        self.experiment_multiselect = widgets.SelectMultiple(
            options=list(DataReader.experiments_mapping.keys()),
            description='Experiments:',
            disabled=False,
            value=[],
        )

        self.experiment_plot = plt.subplots(figsize=(18, 4))
        self.experiment_plot_output = widgets.Output()
         
        def experiment_multiselect_change_callback(change):
            if change.name == 'value' and change.new is not None:
                self.update_experiment_plot(change.new)

        self.experiment_multiselect.observe(experiment_multiselect_change_callback, type='change')
        super(PolicyComparison, self).__init__([self.experiment_multiselect, self.experiment_plot_output])


    def update_experiment_plot(self, experiments):
        with self.experiment_plot_output:
            clear_output()
            self.experiment_plot[1].cla()
            for experiment in experiments:
                steps, result = DataReader.get_success_rates_for_experiment(experiment)
                self.experiment_plot[1].plot(
                    np.array(steps) / 1e3, np.median(result, 0),
                    label=f'{self.experiment_multiselect.value}',
                    linewidth=2,
                )
                self.experiment_plot[1].fill_between(
                    np.array(steps) / 1e3, result.min(0), result.max(0),
                    alpha=.5,
                )
                self.experiment_plot[1].grid(True)
                self.experiment_plot[1].set_xlabel('steps [k–]')
                self.experiment_plot[1].set_ylabel('success rate')
                self.experiment_plot[1].legend()
                self.experiment_plot[1].set_ylim([0.50, 0.85])
                self.experiment_plot[1].set_xlim([5, 105])
                self.experiment_plot[1].set_title(' vs '.join(experiments))
                self.experiment_plot[1].set_xticks(range(10, 100 + 10, 10))
            display(self.experiment_plot[0])

 
class EpisodeReview(widgets.VBox):

    def __init__(self):
        self.episode_play = widgets.Play(
            value=0,
            min=0,
            max=100,
            step=1,
            description="Press play",
            disabled=False
        )
        self.episode_slider = widgets.IntSlider()
        self.episode_hbox = widgets.HBox([self.episode_play, self.episode_slider])

        widgets.jslink((self.episode_play, 'value'), (self.episode_slider, 'value'))
        widgets.jslink((self.episode_play, 'max'), (self.episode_slider, 'max'))
        widgets.jslink((self.episode_play, 'min'), (self.episode_slider, 'min'))

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

        ax_x = bq.Axis(label='time', scale=x_sc, grid_lines='solid')
        ax_x.min = 0
        ax_x.max = 100
        ax_y = bq.Axis(label='costs', scale=y_sc, orientation='vertical', grid_lines='solid')
        ax_y2 = bq.Axis(label='speed', scale=y_sc2, orientation='vertical', side = 'right', grid_lines='none')


        self.costs_plot_lines_costs = Lines(scales={'x': x_sc, 'y': y_sc}, display_legend=True, stroke_width=1)
        self.costs_plot_lines_speed = Lines(scales={'x': x_sc, 'y': y_sc2}, colors=['red'], display_legend=True, stroke_width=1)
        self.costs_plot_progress = Lines(scales={'x': x_sc, 'y': y_sc})
        self.costs_plot_figure = Figure(marks=[self.costs_plot_lines_costs, self.costs_plot_lines_speed, self.costs_plot_progress], 
                                        axes=[ax_x, ax_y, ax_y2],
                                        title='Costs and speed',
                                        legend_location='top-left')


        self.images_hbox = widgets.HBox([self.episode_gradient_image, self.episode_image, self.costs_plot_figure])

        def episode_slider_callback(change):
            if change.name == 'value' and change.new is not None:
                gradient_shift = len(self.images) - len(self.gradient_images)
                if change.new >= gradient_shift:
                    self.episode_gradient_image.value = self.gradient_images[change.new - gradient_shift]
                self.episode_image.value = self.images[change.new]
                self.update_timestamp_line(change.new)

        def pick_episode_callback(experiment, seed, checkpoint, episode):
            self.gradient_images = DataReader.get_gradients(experiment, seed, checkpoint, episode)
            self.images = DataReader.get_images(experiment, seed, checkpoint, episode)

            self.episode_gradient_image.value = self.gradient_images[0]
            self.episode_image.value = self.images[0]
            self.episode_slider.min = 0
            self.episode_slider.value = 0
            self.episode_slider.max = len(self.images)

            self.seed = seed
            self.checkpoint = checkpoint
            self.episode = episode

            self.update_costs_plot(experiment, seed, checkpoint, episode)

        self.picker = ModelPicker(ModelPicker.EPISODE_LEVEL, pick_episode_callback)
        self.episode_slider.observe(episode_slider_callback, type='change')

        self.line = None

        super(EpisodeReview, self).__init__([self.picker, self.episode_hbox, self.images_hbox])

    def update_costs_plot(self, experiment, seed, checkpoint, episode):
        speeds = DataReader.get_episode_speeds(experiment, seed, checkpoint, episode)
        costs = DataReader.get_episode_costs(experiment, seed, checkpoint, episode)
        print(costs.columns)
        x = costs.index
        self.costs_plot_lines_costs.x = x 
        self.costs_plot_lines_costs.y = [costs['proximity_cost'], costs['lane_cost'], costs['pixel_proximity_cost']]
        self.costs_plot_lines_costs.labels=['proximity cost', 'lane cost', 'pixel proximity cost']
        self.costs_plot_lines_speed.x = range(len(speeds))
        self.costs_plot_lines_speed.y = speeds
        self.costs_plot_lines_speed.labels = ['speed']

    def update_timestamp_line(self, timestamp):
        self.costs_plot_progress.x = [timestamp, timestamp]
        self.costs_plot_progress.y = [0, 1]

class PCAPlot(widgets.VBox):

    def __init__(self):
        self.PCA = PCA()

        self.scatter = Scatter(scales={'x': LinearScale(), 'y': LinearScale()})
        self.scatter_figure = Figure(marks=[self.scatter])

        def callback(experiment, seed, step):
            fails = DataReader.get_episodes_with_outcome(experiment, seed, step, 0)
            features = []
            for fail in fails[:3]:
                features.append(PCA.get_episode_features(experiment, seed, step, fail))
            features = np.stack(features)
            res = self.PCA.transform(features)
            print(res.shape)
            self.scatter.x = res[:, 0]
            self.scatter.y = res[:, 1]

        self.picker = ModelPicker(ModelPicker.MODEL_LEVEL, callback)

        super(PCAPlot, self).__init__([self.picker, self.scatter_figure])
