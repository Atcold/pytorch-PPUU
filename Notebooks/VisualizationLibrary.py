import ipywidgets as widgets
from IPython.display import display, clear_output

from glob import glob
import re
import torch
import numpy as np
import pandas
import matplotlib.pyplot as plt
from bqplot.marks import Pie, Bars
from bqplot import Figure
from bqplot.scales import LinearScale

SUCCESS_INDEX_PIE_PLOT = 0
EPISODES = 561

class Visualization:

    def __init__(self):
        self.data_reader = DataReader()
        self.ignore_updates = False

        # ipywidget definitions
        self.select_experiment = widgets.Select(
            options=list(self.data_reader.experiments_mapping.keys()),
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
        self.clean_button = widgets.Button(description="Clean the plot")
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
        self.bars_plot = Bars(x = np.arange(10), y = np.arange(10), scales={'x':LinearScale(), 'y':LinearScale()})
        self.episode_play = widgets.Play(
            value=0,
            min=0,
            max=100,
            step=1,
            description="Press play",
            disabled=False
        )

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
        self.bars_figure = Figure(title='Success rate per episode', marks=[self.bars_plot], layout=widgets.Layout(width='100%', height='300'))
        plt.ioff()
        self.experiment_plot = plt.subplots(figsize=(18, 4))
        self.experiment_plot_output = widgets.Output()

        self.costs_plot = plt.subplots(figsize=(18, 4))
        self.costs_plot = *self.costs_plot, self.costs_plot[1].twinx()
        self.costs_plot_output = widgets.Output()

        self.success_matrix = plt.subplots(figsize=(18, 4))
        self.success_matrix_output = widgets.Output()

        # callbacks definitions
        # They're defined in the initializer so that we have access to self.
        def select_experiment_change_callback(change):
            if change.name == 'value' and change.new is not None:
                self.ignore_updates = True
                self.seed_dropdown.options = self.data_reader.find_option_values('seed', 
                                                                     experiment=self.select_experiment.value)
                self.seed_dropdown.value = None
                self.checkpoint_dropdown.value = None
                self.ignore_updates = False

                self.update_bars_plot()
                self.update_experiment_plot()

        def seed_dropdown_change_callback(change):
            if self.ignore_updates:
                return
            if change.name == 'value' and change.new is not None:
                self.ignore_updates = True
                self.checkpoint_dropdown.options = self.data_reader.find_option_values('checkpoint', 
                                                                           experiment=self.select_experiment.value,
                                                                           seed=self.seed_dropdown.value)
                self.checkpoint_dropdown.value = None
                self.ignore_updates = False

        def checkpoint_dropdown_change_callback(change):
            if self.ignore_updates:
                return
            if change.name == 'value' and change.new is not None:
                self.ignore_updates = True
                self.episode_dropdown.options = self.data_reader.find_option_values('episode', 
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

        def clean_button_click_callback(b):
            with self.experiment_plot_output:
                clear_output()
                self.experiment_plot[1].cla()
                display(self.experiment_plot[0])

        def episode_slider_callback(change):
            if change.name == 'value' and change.new is not None:
                self.episode_gradient_image.value = self.gradient_images[change.new]
                self.episode_image.value = self.images[change.new]

        self.select_experiment.observe(select_experiment_change_callback, type='change')
        self.seed_dropdown.observe(seed_dropdown_change_callback, type='change')
        self.checkpoint_dropdown.observe(checkpoint_dropdown_change_callback, type='change')
        self.clean_button.on_click(clean_button_click_callback)
        self.episode_dropdown.observe(episode_dropdown_change_callback ,type='change')
        self.episode_slider.observe(episode_slider_callback, type='change')

        # layout
        self.experiment_plot_box = widgets.VBox([self.experiment_plot_output, self.clean_button])
        self.episode_images_box = widgets.VBox([self.episode_hbox, self.images_hbox])

        self.tab = widgets.Tab()
        self.tab.children = [self.experiment_plot_box, self.episode_images_box, self.pie_figure, self.bars_figure, 
                self.costs_plot_output, self.success_matrix_output]
        titles = ['Policy performance', 'Episode review', 'Success Pie', 'Success Bars', 'Costs', 'Success Matrix']
        for i in range(len(self.tab.children)):
            self.tab.set_title(i, titles[i])


    def update_bars_plot(self):
        if self.select_experiment.value is None:
            return
        result = self.data_reader.get_episodes_success_counts(self.select_experiment.value)
        self.bars_plot.y = result
        self.bars_plot.x = np.arange(len(result))
        with self.success_matrix_output:
            clear_output()
            self.success_matrix[1].matshow(result.reshape(11, 51))
            display(self.success_matrix[0])


    def update_experiment_plot(self):
        steps, result = self.data_reader.get_success_rates_for_experiment(self.select_experiment.value)
        with self.experiment_plot_output:
            clear_output()
            self.experiment_plot[1].plot(
                np.array(steps) / 1e3, np.median(result, 0),
                label=f'{self.select_experiment.value}',
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
            self.experiment_plot[1].set_title('Regressed vs. hardwired cost policy success rate min-max')
            self.experiment_plot[1].set_xticks(range(10, 100 + 10, 10))
            display(self.experiment_plot[0])


    def update_pie_chart(self):
        success_rate = self.data_reader.get_success_rate(self.select_experiment.value,
                                                         self.seed_dropdown.value,
                                                         self.checkpoint_dropdown.value)
        self.pie_plot.sizes = [success_rate, 1 - success_rate]
        self.pie_plot.labels = [str(round(success_rate, 2)), str(round(1 - success_rate, 2))]


    def update_costs_plot(self):
        speeds = self.data_reader.get_episode_speeds(self.select_experiment.value, 
                                    self.seed_dropdown.value,
                                    self.checkpoint_dropdown.value,
                                    self.episode_dropdown.value)
        costs = self.data_reader.get_episode_costs(self.select_experiment.value, 
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
            N = 50
            costs.tail(N).plot(ax=self.costs_plot[1])
            self.costs_plot[1].legend(loc='upper center', ncol=4)
            self.costs_plot[1].grid(True)
            display(self.costs_plot[0])


    def update_episode_viewer(self):
        self.gradient_images = self.data_reader.get_gradients(
                                                     self.select_experiment.value,
                                                     self.seed_dropdown.value,
                                                     self.checkpoint_dropdown.value,
                                                     self.episode_dropdown.value
                                                 )
        self.images = self.data_reader.get_images(self.select_experiment.value,
                                      self.seed_dropdown.value,
                                      self.checkpoint_dropdown.value,
                                      self.episode_dropdown.value)

        self.episode_gradient_image.value = self.gradient_images[0]
        self.episode_image.value = self.images[0]
        self.episode_slider.min = 0
        self.episode_slider.value = 0
        self.episode_slider.max = len(self.gradient_images)


    def display(self):
        display(self.select_experiment)
        display(self.seed_dropdown)
        display(self.checkpoint_dropdown)
        display(self.episode_dropdown)
        display(self.tab)


class DataReader:

    experiments_mapping = {
        'Stochastic': [
            '/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v12/',
            'MPUR-policy-gauss-model=vae-zdropout=0.5-policy-gauss-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=1',
        ],

        'Deterministic policy, regressed cost': [
            '/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v12/',
            'MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=1',
        ],

        'Non-regressed cost' : [
            '/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v13/',
            'MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=False',
        ],
    }

    def __init__(self):
        pass

    def get_images(self, experiment, seed, checkpoint, episode):
        path = self.experiments_mapping[experiment][0]
        model_name = self.experiments_mapping[experiment][1]
        # gradient_path = f'{path}/planning_results/grad_videos_simulator/{model_name}-seed={seed}-novaluestep-{checkpoint}.model/'
        image_paths = f'/misc/vlgscratch4/LecunGroup/nvidia-collab/vlad/models/eval_with_cost/planning_results/videos_simulator/MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=1-seed=3-novaluestep25000.model/ep{episode}/ego/*.png'
        images = []
        for image_path in sorted(glob(image_paths)):
            with open(image_path, 'rb') as f:
                images.append(f.read())
        return images

    def get_gradients(self, experiment, seed, checkpoint, episode):
        path = self.experiments_mapping[experiment][0]
        model_name = self.experiments_mapping[experiment][1]
        # gradient_path = f'{path}/planning_results/grad_videos_simulator/{model_name}-seed={seed}-novaluestep-{checkpoint}.model/'
        gradient_paths = f'/misc/vlgscratch4/LecunGroup/nvidia-collab/vlad/models/eval_with_cost/planning_results/grad_videos_simulator/MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=1-seed=3-novaluestep25000.model/ep{episode}/*.png'
        images = []
        for image_path in sorted(glob(gradient_paths)):
            with open(image_path, 'rb') as f:
                images.append(f.read())
        return images

    def get_evaluation_log_file(self, experiment, seed, step):
        path = self.experiments_mapping[experiment]
        regex = path[0] + 'planning_results/' + path[1] + f'-seed={seed}-novaluestep{step}' + '.model.log'
        paths = glob(regex)
        assert len(paths) == 1, f'paths for {regex} is not length of 1, and is equal to {paths}'
        return paths[0]

    def find_option_values(self, option, experiment=None, seed=None, checkpoint=None):
        if option == 'seed':
            path = self.experiments_mapping[experiment]
            logs = glob(path[0] + 'policy_networks/' + path[1] + '*.log')
            regexp = r"seed=(\d+)-"
        elif option == 'checkpoint':
            path = self.experiments_mapping[experiment]
            logs = glob(path[0] + 'planning_results/' + path[1] + f'-seed={seed}' + '*.model.log')
            regexp = r'-novaluestep(\d+)\.'
        elif option == 'episode':
            path = self.experiments_mapping[experiment]
            logs = glob(path[0] + 'planning_results/videos_simulator/' + path[1] + f'-seed={seed}-novaluestep{checkpoint}.model/ep*')
            regexp = r'model/ep(\d+)'

        values = []

        for log in logs:
            m = re.search(regexp, log)
            result = m.group(1)
            values.append(int(result))

        values.sort()

        return values

    def get_success_rate(self, experiment, seed, step):
        with open(self.get_evaluation_log_file(experiment, seed, step), 'r') as f:
            last_line = f.readlines()[-1]
            last_colon = last_line.rfind(':')
            success_rate = float(last_line[(last_colon + 2):])
        return success_rate

    def get_success_rates_for_experiment(self, experiment):
        seeds = self.find_option_values('seed', experiment)
        result = {} 
        steps = []
        for seed in seeds:
            result[seed] = []
            checkpoints = self.find_option_values('checkpoint', experiment, seed)
            if len(steps) < len(checkpoints):
                steps = checkpoints

            for checkpoint in checkpoints:
                success = self.get_success_rate(experiment, seed, checkpoint)
                result[seed].append(success)

        result = np.stack([np.array(result[seed]) for seed in result])
        steps = np.array(steps)
        return steps, result

    def get_episodes_with_outcome(self, experiment, seed, step, outcome):
        path = self.get_evaluation_log_file(experiment, seed, step)
        with open(path, 'r') as f:
            lines = f.readlines()
        regex = re.compile(".*ep:\s+(\d+).*\|\ssuccess:\s+(\d).*")
        result = []
        for line in lines:
            match = regex.match(line)
            if match:
                if int(match.group(2)) == outcome:
                    result.append(int(match.group(1)))
        return result

    def get_episodes_success_counts(self, experiment):
        seeds = self.find_option_values('seed', experiment)
        result = np.zeros(EPISODES)
        for seed in seeds:
            checkpoints = self.find_option_values('checkpoint', experiment, seed)
            for checkpoint in checkpoints:
                success = self.get_episodes_with_outcome(experiment,
                                                         seed, 
                                                         checkpoint,
                                                         1)
                success = np.array(success)
                success = success - 1
                one_hot = np.zeros((len(success), EPISODES))
                one_hot[np.arange(len(success)), success] = 1
                one_hot = np.sum(one_hot, axis=0),
                one_hot = np.squeeze(one_hot)
                result += one_hot
        return result

    def get_episode_speeds(self, experiment, seed, checkpoint, episode):
        path = self.experiments_mapping[experiment]
        regex = path[0] + 'planning_results/' + path[1] + f'-seed={seed}-novaluestep{checkpoint}' + '.model.states'
        states_paths = glob(regex)
        assert len(states_paths) == 1, f'states_paths for {regex} is {states_paths} and it\'s length is not 1'
        states_path = states_paths[0]
        states = torch.load(states_path)
        episode_states = states[episode - 1]
        episode_states = list(map(lambda x : x[-1], episode_states))
        episode_states = torch.stack(episode_states)
        episode_states[:, 2:].norm(dim=1)
        return episode_states[:, 2:].norm(dim=1) # is it correct

    def get_episode_costs(self, experiment, seed, checkpoint, episode):
        """ Returns an array of data frames with all the costs for given evaluation """
        file_name = '/misc/vlgscratch4/LecunGroup/nvidia-collab/vlad/models/eval_with_cost/planning_results/MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=1-seed=3-novaluestep25000.model.costs'
        raw_costs = torch.load(file_name)
        costs = [pandas.DataFrame(c) for c in raw_costs]  # list of DataFrame, one per episode
        return costs[episode]
