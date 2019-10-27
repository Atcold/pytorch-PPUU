import numpy as np
from glob import glob
import torch
import pandas
import re

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

    @staticmethod
    def get_images(experiment, seed, checkpoint, episode):
        path = DataReader.experiments_mapping[experiment][0]
        model_name = DataReader.experiments_mapping[experiment][1]
        # gradient_path = f'{path}/planning_results/grad_videos_simulator/{model_name}-seed={seed}-novaluestep-{checkpoint}.model/'
        image_paths = f'/misc/vlgscratch4/LecunGroup/nvidia-collab/vlad/models/eval_with_cost/planning_results/videos_simulator/MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=1-seed=3-novaluestep25000.model/ep{episode}/ego/*.png'
        # image_paths = f'/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v13/planning_results/videos_simulator/MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=False-seed=1-novaluestep70000.model/ep{episode}/ego/*.png'
        images = []
        for image_path in sorted(glob(image_paths)):
            with open(image_path, 'rb') as f:
                images.append(f.read())
        print('images length is', len(images))
        return images

    @staticmethod
    def get_gradients(experiment, seed, checkpoint, episode):
        path = DataReader.experiments_mapping[experiment][0]
        model_name = DataReader.experiments_mapping[experiment][1]
        # gradient_path = f'{path}/planning_results/grad_videos_simulator/{model_name}-seed={seed}-novaluestep-{checkpoint}.model/'
        gradient_paths = f'/misc/vlgscratch4/LecunGroup/nvidia-collab/vlad/models/eval_with_cost/planning_results/grad_videos_simulator/MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=1-seed=3-novaluestep25000.model/ep{episode}/*.png'
        images = []
        for image_path in sorted(glob(gradient_paths)):
            with open(image_path, 'rb') as f:
                images.append(f.read())
        return images

    @staticmethod
    def get_evaluation_log_file(experiment, seed, step):
        path = DataReader.experiments_mapping[experiment]
        regex = path[0] + 'planning_results/' + path[1] + f'-seed={seed}-novaluestep{step}' + '.model.log'
        paths = glob(regex)
        assert len(paths) == 1, f'paths for {regex} is not length of 1, and is equal to {paths}'
        return paths[0]

    @staticmethod
    def find_option_values(option, experiment=None, seed=None, checkpoint=None):
        if option == 'seed':
            path = DataReader.experiments_mapping[experiment]
            logs = glob(path[0] + 'policy_networks/' + path[1] + '*.log')
            regexp = r"seed=(\d+)-"
        elif option == 'checkpoint':
            path = DataReader.experiments_mapping[experiment]
            logs = glob(path[0] + 'planning_results/' + path[1] + f'-seed={seed}' + '*.model.log')
            regexp = r'-novaluestep(\d+)\.'
        elif option == 'episode':
            path = DataReader.experiments_mapping[experiment]
            logs = glob(path[0] + 'planning_results/videos_simulator/' + path[1] + f'-seed={seed}-novaluestep{checkpoint}.model/ep*')
            regexp = r'model/ep(\d+)'

        values = []

        for log in logs:
            m = re.search(regexp, log)
            result = m.group(1)
            values.append(int(result))

        values.sort()

        return values

    @staticmethod
    def get_success_rate(experiment, seed, step):
        with open(DataReader.get_evaluation_log_file(experiment, seed, step), 'r') as f:
            last_line = f.readlines()[-1]
            last_colon = last_line.rfind(':')
            success_rate = float(last_line[(last_colon + 2):])
        return success_rate

    @staticmethod
    def get_success_rates_for_experiment(experiment):
        seeds = DataReader.find_option_values('seed', experiment)
        result = {} 
        steps = []
        for seed in seeds:
            result[seed] = []
            checkpoints = DataReader.find_option_values('checkpoint', experiment, seed)
            if len(steps) < len(checkpoints):
                steps = checkpoints

            for checkpoint in checkpoints:
                success = DataReader.get_success_rate(experiment, seed, checkpoint)
                result[seed].append(success)

        result = np.stack([np.array(result[seed]) for seed in result])
        steps = np.array(steps)
        return steps, result

    @staticmethod
    def get_episodes_with_outcome(experiment, seed, step, outcome):
        path = DataReader.get_evaluation_log_file(experiment, seed, step)
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

    @staticmethod
    def get_episodes_success_counts(experiment):
        seeds = DataReader.find_option_values('seed', experiment)
        result = np.zeros(EPISODES)
        for seed in seeds:
            checkpoints = DataReader.find_option_values('checkpoint', experiment, seed)
            for checkpoint in checkpoints:
                success = DataReader.get_episodes_with_outcome(experiment,
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

    @staticmethod
    def get_episode_speeds(experiment, seed, checkpoint, episode):
        path = DataReader.experiments_mapping[experiment]
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

    @staticmethod
    def get_episode_costs(experiment, seed, checkpoint, episode):
        """ Returns an array of data frames with all the costs for given evaluation """
        file_name = '/misc/vlgscratch4/LecunGroup/nvidia-collab/vlad/models/eval_with_cost/planning_results/MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=1-seed=3-novaluestep25000.model.costs'
        raw_costs = torch.load(file_name)
        costs = pandas.DataFrame(raw_costs[episode])  # list of DataFrame, one per episode
        return costs
