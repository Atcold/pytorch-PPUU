import ipywidgets as widgets
from IPython.display import display

from glob import glob
import re
import torch
import matplotlib.pyplot as plt
from bqplot.marks import Pie
from bqplot import Figure

SUCCESS_INDEX_PIE_PLOT = 0

class Visualization:

    policies_mapping = {
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
        self.ignore_udpates = False

        self.select_experiment = widgets.Select(
            options=list(self.policies_mapping.keys()),
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

        self.pie_plot = Pie(radius=150,
                inner_radius=80,
                interactions={'click': 'select'},
                colors=['green', 'red'],
                label_color='black',
                font_size='14px',
                )

        self.figure = Figure(title='Success rate', marks=[self.pie_plot])

        # self.episode_dropdown = widgets.Dropdown(
        #     description='Episode:',
        #     disabled=False,
        # )

        def select_experiment_change_callback(change):
            if change.name == 'value' and change.new is not None:
                self.ignore_udpates = True
                self.seed_dropdown.options = self.find_option_values('seed')
                self.seed_dropdown.value = None
                self.checkpoint_dropdown.value = None
                self.ignore_udpates = False

        def seed_dropdown_change_callback(change):
            if self.ignore_udpates:
                return
            if change.name == 'value' and change.new is not None:
                self.ignore_udpates = True
                self.checkpoint_dropdown.options = self.find_option_values('checkpoint')
                self.checkpoint_dropdown.value = None
                self.ignore_udpates = False

        def checkpoint_dropdown_change_callback(change):
            if self.ignore_udpates:
                return
            # self.episode_dropdown.options = self.find_option_values('episode')
            if change.name == 'value' and change.new is not None:
                self.get_success_rate(self.select_experiment.value, self.seed_dropdown.value, self.checkpoint_dropdown.value)

        def plot_episode_state(episode):
            if self.episode_dropdown.value is not None:
                path = self.policies_mapping[self.select_experiment.value]
                regex = path[0] + 'planning_results/' + path[1] + f'-seed={self.seed_dropdown.value}-novaluestep{self.checkpoint_dropdown.value}' + '.model.states'
                states_paths = glob(regex)
                assert len(states_paths) == 1, f'states_paths for {regex} is {states_paths} and it\'s length is not 1'
                states_path = states_paths[0]
                states = torch.load(states_path)
                print('value is ', self.episode_dropdown.value)
                print('states len is', len(states))
                episode_states = states[self.episode_dropdown.value - 1]
                episode_states = list(map(lambda x : x[-1], episode_states))
                episode_states = torch.stack(episode_states)
                episode_states[:, 2:].norm(dim=1)
                plt.plot(episode_states[:, 2:].norm(dim=1))
                plt.show()

        def pie_plot_click_callback(x, y):
            if y['data']['index'] == SUCCESS_INDEX_PIE_PLOT:
                # failure cases
                print(self.get_episodes_with_outcome(self.select_experiment.value, self.seed_dropdown.value, self.checkpoint_dropdown.value, 1))
            else:
                # success cases
                print(self.get_episodes_with_outcome(self.select_experiment.value, self.seed_dropdown.value, self.checkpoint_dropdown.value, 0))

        self.select_experiment.observe(select_experiment_change_callback, type='change')
        self.seed_dropdown.observe(seed_dropdown_change_callback, type='change')
        self.checkpoint_dropdown.observe(checkpoint_dropdown_change_callback, type='change')
        self.pie_plot.on_element_click(pie_plot_click_callback)
        # self.episode_dropdown.observe(plot_episode_state,type='change')

    def find_option_values(self, option=None, option_arg=0):
    #     ipdb.set_trace()
        if option == 'seed':
            path = self.policies_mapping[self.select_experiment.value]
            logs = glob(path[0] + 'policy_networks/' + path[1] + '*.log')
            regexp = r"seed=(\d+)-"
        elif option == 'checkpoint':
            path = self.policies_mapping[self.select_experiment.value]
            logs = glob(path[0] + 'planning_results/' + path[1] + f'-seed={self.seed_dropdown.value}' + '*.model.log')
            regexp = r'-novaluestep(\d+)\.'
        elif option == 'episode':
            path = self.policies_mapping[self.select_experiment.value]
            logs = glob(path[0] + 'planning_results/videos_simulator/' + path[1] + f'-seed={self.seed_dropdown.value}-novaluestep{self.checkpoint_dropdown.value}.model/ep*')
            regexp = r'model/ep(\d+)'

        values = []

        for log in logs:
            m = re.search(regexp, log)
            result = m.group(1)
            values.append(int(result))

        values.sort()

        return values

    def get_evaluation_log_file(self, policy, seed, step):
        path = self.policies_mapping[policy]
        regex = path[0] + 'planning_results/' + path[1] + f'-seed={seed}-novaluestep{step}' + '.model.log'
        paths = glob(regex)
        assert len(paths) == 1, f'paths for {regex} is not length of 1, and is equal to {paths}'
        return paths[0]

    def get_success_rate(self, policy, seed, step):
        with open(self.get_evaluation_log_file(policy, seed, step), 'r') as f:
            last_line = f.readlines()[-1]
            last_colon = last_line.rfind(':')
            success_rate = float(last_line[(last_colon + 2):])
        self.pie_plot.sizes = [success_rate, 1 - success_rate]
        self.pie_plot.labels = [str(round(success_rate, 2)), str(round(1 - success_rate, 2))]

    def get_episodes_with_outcome(self, policy, seed, step, outcome):
        path = self.get_evaluation_log_file(policy, seed, step)
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
    
    def display(self):
        display(self.select_experiment)
        display(self.seed_dropdown)
        display(self.checkpoint_dropdown)
        display(self.figure)
        # display(self.episode_dropdown)
