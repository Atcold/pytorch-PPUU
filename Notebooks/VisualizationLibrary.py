import ipywidgets as widgets
from IPython.display import display, clear_output

from glob import glob
import re
import torch
import numpy as np
import pandas
import matplotlib.pyplot as plt
from bqplot.marks import Pie, GridHeatMap, Label, Lines, Bars
from bqplot import Figure
from bqplot.scales import LinearScale, ColorScale, OrdinalScale
import bqplot as bq

from DataReader import DataReader
from Widgets import ModelPicker, EpisodeReview, PolicyComparison, PCAPlot

SUCCESS_INDEX_PIE_PLOT = 0
EPISODES = 561

plt.ioff()

class Visualization:

    def __init__(self):
        self.ignore_updates = False

        # ipywidget definitions
        self.select_experiment = widgets.Select(
            options=list(DataReader.experiments_mapping.keys()),
            rows=10,
            description='Experiments',
            disabled=False,
            layout=widgets.Layout(width='500px',border='solid'),
        )
        self.seed_dropdown = widgets.Dropdown(
            description='Seed:',
            disabled=False,
        )
        self.checkpoint_dropdown = widgets.Dropdown(
            description='Checkpoint:',
            disabled=False,
        )

        self.episode_dropdown = widgets.Dropdown(
            description='Successful Episode:',
            disabled=False,
        )

        # plots definitions
        self.pie_plot = Pie(radius=150,
                inner_radius=80,
                interactions={'click': 'select'},
                colors=['green', 'red'],
                label_color='black',
                font_size='14px',
                )
        self.episode_play = widgets.Play(
            value=0,
            min=0,
            max=100,
            step=1,
            description="Press play",
            disabled=False
        )
        # self.episode_grid_heat_map = GridHeatMap(scales={'row': LinearScale(), 'column': LinearScale(), 'color': ColorScale()}, color=np.(1, 1))
        self.episode_grid_heat_map = GridHeatMap(color=np.random.rand(11, 51) * 0, scales={'row': OrdinalScale(), 'column': OrdinalScale(), 'color': ColorScale()}, display_legend=False)
        self.episode_grid_heat_map_label = widgets.Label()

        self.episode_slider = widgets.IntSlider()
        widgets.jslink((self.episode_play, 'value'), (self.episode_slider, 'value'))
        widgets.jslink((self.episode_play, 'max'), (self.episode_slider, 'max'))
        widgets.jslink((self.episode_play, 'min'), (self.episode_slider, 'min'))
        self.episode_hbox = widgets.HBox([self.episode_play, self.episode_slider])
        self.episode_gradient_image = widgets.Image(format='png',
                                                    width=120,
                                                    height=600,
                                                    )
        self.episode_image = widgets.Image(format='png',
                                           width=120,
                                           height=600,
                                           )
        self.images_hbox = widgets.HBox([self.episode_gradient_image, self.episode_image])

        # figures containing plots definition
        self.pie_figure = Figure(title='Success rate', marks=[self.pie_plot])
        self.episode_grid_heat_map_figure = Figure(title='Episode grid heat map', marks=[self.episode_grid_heat_map], layout=widgets.Layout(height='330px', width='100%'))

        self.costs_plot = plt.subplots(figsize=(18, 4))
        self.costs_plot = *self.costs_plot, self.costs_plot[1].twinx()
        self.costs_plot_output = widgets.Output()


        # callbacks definitions
        # They're defined in the initializer so that we have access to self.
        def select_experiment_change_callback(change):
            if change.name == 'value' and change.new is not None:
                self.ignore_updates = True
                self.seed_dropdown.options = DataReader.find_option_values(option='seed', 
                                                                           experiment=self.select_experiment.value)
                self.seed_dropdown.value = None
                self.checkpoint_dropdown.value = None
                self.ignore_updates = False
                self.update_heatmap()

        def seed_dropdown_change_callback(change):
            if self.ignore_updates:
                return
            if change.name == 'value' and change.new is not None:
                self.ignore_updates = True
                self.checkpoint_dropdown.options = DataReader.find_option_values(option='checkpoint', 
                                                                           experiment=self.select_experiment.value,
                                                                           seed=self.seed_dropdown.value)
                self.checkpoint_dropdown.value = None
                self.ignore_updates = False

        def checkpoint_dropdown_change_callback(change):
            if self.ignore_updates:
                return
            if change.name == 'value' and change.new is not None:
                self.ignore_updates = True
                self.episode_dropdown.options = DataReader.find_option_values(option='episode', 
                                                                        experiment=self.select_experiment.value,
                                                                        seed=self.seed_dropdown.value,
                                                                        checkpoint=self.checkpoint_dropdown.value)
                self.episode_dropdown.value = None
                self.ignore_updates = False
                self.update_pie_chart()

        def episode_dropdown_change_callback(change):
            if self.ignore_updates:
                return
            if change.name == 'value' and change.new is not None:
                self.update_episode_viewer()
                self.update_costs_plot()

        def episode_slider_callback(change):
            if change.name == 'value' and change.new is not None:
                gradient_shift = len(self.images) - len(self.gradient_images)
                if change.new >= gradient_shift:
                    self.episode_gradient_image.value = self.gradient_images[change.new - gradient_shift]
                self.episode_image.value = self.images[change.new]

        def heat_map_click_callback(a, b):
            if self.result_permutation is not None:
                episode = self.result_permutation[b['data']['_cell_num']] + 1
                color = b['data']['color']
                self.episode_grid_heat_map_label.value = f'clicked on episode {episode} with {color} successful cases'

        self.select_experiment.observe(select_experiment_change_callback, type='change')
        self.seed_dropdown.observe(seed_dropdown_change_callback, type='change')
        self.checkpoint_dropdown.observe(checkpoint_dropdown_change_callback, type='change')
        self.episode_dropdown.observe(episode_dropdown_change_callback ,type='change')
        self.episode_slider.observe(episode_slider_callback, type='change')

        self.episode_grid_heat_map.on_click(heat_map_click_callback)
        self.episode_grid_heat_map.on_element_click(heat_map_click_callback)

        # layout
        self.episode_images_box = widgets.VBox([self.episode_hbox, self.images_hbox])
        self.heatmap_box = widgets.VBox([self.episode_grid_heat_map_figure, self.episode_grid_heat_map_label])

        def foo(*argv):
            print('args ', argv)

        x = ModelPicker(ModelPicker.EXPERIMENT_LEVEL, foo)

        self.tab = widgets.Tab()
        # self.tab.children = [PolicyComparison(), self.episode_images_box, self.pie_figure, 
        #         self.costs_plot_output, self.heatmap_box, EpisodeReview()]
        # titles = ['Policy performance', 'Episode review', 'Success Pie', 'Costs', 'Success Heatmap', 'test']
        self.tab.children = [PCAPlot()]
        titles = ['pca']
        for i in range(len(self.tab.children)):
            self.tab.set_title(i, titles[i])


    def update_heatmap(self):
        if self.select_experiment.value is None:
            return
        result = DataReader.get_episodes_success_counts(self.select_experiment.value)
        self.result_permutation = np.argsort(result)
        result = np.sort(result)
        self.episode_grid_heat_map.color = result.reshape(11, 51)
        self.color_result = result


    def update_pie_chart(self):
        success_rate = DataReader.get_success_rate(self.select_experiment.value,
                                                         self.seed_dropdown.value,
                                                         self.checkpoint_dropdown.value)
        self.pie_plot.sizes = [success_rate, 1 - success_rate]
        self.pie_plot.labels = [str(round(success_rate, 2)), str(round(1 - success_rate, 2))]


    def update_costs_plot(self):
        speeds = DataReader.get_episode_speeds(self.select_experiment.value, 
                                    self.seed_dropdown.value,
                                    self.checkpoint_dropdown.value,
                                    self.episode_dropdown.value)
        costs = DataReader.get_episode_costs(self.select_experiment.value, 
                                    self.seed_dropdown.value,
                                    self.checkpoint_dropdown.value,
                                    self.episode_dropdown.value)
        with self.costs_plot_output:
            clear_output()
            self.costs_plot[1].cla()
            self.costs_plot[2].cla()
            self.costs_plot[1].set_ylim([-0.1, 1.1])
            self.costs_plot[2].plot(speeds, color='C4')
            self.costs_plot[2].set_ylabel('speed', color='C4')
            self.costs_plot[2].tick_params(axis='y', labelcolor='C4')
            display(self.costs_plot[2])
            costs.tail(N).plot(ax=self.costs_plot[1])
            self.costs_plot[1].legend(loc='upper center', ncol=4)
            self.costs_plot[1].grid(True)
            display(self.costs_plot[0])


    def update_episode_viewer(self):
        self.gradient_images = DataReader.get_gradients(
                                                     self.select_experiment.value,
                                                     self.seed_dropdown.value,
                                                     self.checkpoint_dropdown.value,
                                                     self.episode_dropdown.value
                                                 )
        self.images = DataReader.get_images(self.select_experiment.value,
                                      self.seed_dropdown.value,
                                      self.checkpoint_dropdown.value,
                                      self.episode_dropdown.value)

        self.episode_gradient_image.value = self.gradient_images[0]
        self.episode_image.value = self.images[0]
        self.episode_slider.min = 0
        self.episode_slider.value = 0
        self.episode_slider.max = len(self.images)


    def display(self):
        display(self.select_experiment)
        display(self.seed_dropdown)
        display(self.checkpoint_dropdown)
        display(self.episode_dropdown)
        display(self.tab)
