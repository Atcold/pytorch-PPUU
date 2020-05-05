"""A class with static methods which can be used to access the data about
experiments.
This includes reading logs to parse success cases, reading images, costs
and speed.
"""

import numpy as np
from glob import glob
import torch
import pandas
import re
import json
from functools import lru_cache
import imageio


EPISODES = 561


class DataReader:
    """Container class for the static data access methods"""

    EXPERIMENTS_MAPPING_FILE = 'experiments_mapping.json'

    @staticmethod
    @lru_cache(maxsize=1)
    def get_experiments_mapping():
        """Reads the experiments mapping from a json file
        EXPERIMENTS_MAPPING_FILE
        """
        with open(DataReader.EXPERIMENTS_MAPPING_FILE, 'r') as f:
            x = json.load(f)
        return x

    @staticmethod
    def get_images(experiment, seed, checkpoint, episode):
        """Get simulator images for a given model evaluation on a 
        given episode"""
        path = DataReader.get_experiments_mapping()[experiment][0]
        model_name = DataReader.get_experiments_mapping()[experiment][1]
        image_paths = f'{path}/planning_results/videos_simulator/{model_name}-seed={seed}-novaluestep{checkpoint}.model/ep{episode}/ego/*.png'
        images = []
        for image_path in sorted(glob(image_paths)):
            with open(image_path, 'rb') as f:
                images.append(f.read())
        return images

    @staticmethod
    def get_gradients(experiment, seed, checkpoint, episode):
        """Get gradients for a given model evaluation on a given episode"""
        path = DataReader.get_experiments_mapping()[experiment][0]
        model_name = DataReader.get_experiments_mapping()[experiment][1]
        gradient_paths = f'{path}/planning_results/grad_videos_simulator/{model_name}-seed={seed}-novaluestep{checkpoint}.model/ep{episode}/*.png'
        images = []
        for image_path in sorted(glob(gradient_paths)):
            with open(image_path, 'rb') as f:
                images.append(f.read())
        return images

    @staticmethod
    def get_last_gradient(experiment, seed, checkpoint, episode):
        """Get the last gradient for the model and episode

        Returns:
            (value, x, y) - tuple, where value is the max value of the
                            gradient, x, y are the location of this max
                            value in the  gradient image.
        """
        path = DataReader.get_experiments_mapping()[experiment][0]
        model_name = DataReader.get_experiments_mapping()[experiment][1]
        gradient_paths = f'{path}/planning_results/grad_videos_simulator/{model_name}-seed={seed}-novaluestep{checkpoint}.model/ep{episode}/*.png'
        images = sorted(glob(gradient_paths))
        if len(images) == 0:
            return (0, 0, 0)
        image_path = sorted(glob(gradient_paths))[-1]
        image = imageio.imread(image_path)
        mx_index = np.argmax(image)
        value = image.flatten()[mx_index]
        middle_x = image.shape[0] / 2
        middle_y = image.shape[1] / 2
        x = mx_index // image.shape[1]
        x -= middle_x
        y = mx_index % image.shape[1]
        y -= middle_y
        if value == 0:
            return (0, 0, 0)
        else:
            return (value, x, y)

    @staticmethod
    def get_evaluation_log_file(experiment, seed, step):
        """Retuns a path to the eval logs for given model"""
        path = DataReader.get_experiments_mapping()[experiment]
        regex = path[0] + 'planning_results/' + path[1] + \
            f'-seed={seed}-novaluestep{step}' + '.model.log'
        paths = glob(regex)
        assert len(paths) == 1, \
            f'paths for {regex} is not length of 1, and is equal to {paths}'
        return paths[0]

    @staticmethod
    def get_training_log_file(experiment, seed):
        """Retuns a path to the eval logs for given model"""
        path = DataReader.get_experiments_mapping()[experiment]
        regex = path[0] + 'policy_networks/' + path[1] + \
            f'-seed={seed}-novalue' + '.log'
        paths = glob(regex)
        assert len(paths) == 1, \
            f'paths for {regex} is not length of 1, and is equal to {paths}'
        return paths[0]

    @staticmethod
    @lru_cache(maxsize=100)
    def find_option_values(option,
                           experiment=None,
                           seed=None,
                           checkpoint=None):
        """Returns possible values for selected option.
        Depending on option, returns:
            if option == 'seed' - returns all seeds for given experiment.
                                  experiment has to passed.
            if option == 'checkpoint' - returns all checkpoints for given
                                        experiment and seed.
                                        experiment and seed have to be
                                        passed.
            if option == 'episode' - returns all episodes for given
                                        model
                                        experiment, seed, and checkpoint have
                                        to be passed.
        """
        if option == 'seed':
            path = DataReader.get_experiments_mapping()[experiment]
            logs = glob(path[0] + 'planning_results/' + path[1] + '*.log')
            regexp = r"seed=(\d+)-"
        elif option == 'checkpoint':
            path = DataReader.get_experiments_mapping()[experiment]
            logs = glob(path[0] + 'planning_results/' +
                        path[1] + f'-seed={seed}' + '*.model.log')
            regexp = r'-novaluestep(\d+)\.'
        elif option == 'episode':
            path = DataReader.get_experiments_mapping()[experiment]
            logs = glob(path[0] +
                        'planning_results/videos_simulator/' +
                        path[1] +
                        f'-seed={seed}-novaluestep{checkpoint}.model/ep*')
            regexp = r'model/ep(\d+)'

        values = []

        for log in logs:
            m = re.search(regexp, log)
            if m:
                result = m.group(1)
                values.append(int(result))
            else:
                print(f'{log} doesn\'t contain {option}')

        # log files for each step are generated for seeds
        values = list(set(values))
        values.sort()

        return values

    @staticmethod
    def get_success_rate(experiment, seed, step):
        """get the success rate for a given model"""
        log_file = DataReader.get_evaluation_log_file(experiment, seed, step)
        with open(log_file, 'r') as f:
            last_line = f.readlines()[-1]
            last_colon = last_line.rfind(':')
            success_rate = float(last_line[(last_colon + 2):])
        return success_rate

    @staticmethod
    def get_success_rates_for_experiment(experiment):
        """get success rate arrays for each seed for the given experiment
        across all checkpoints.
        The resulting shape of the np array is
        (seeds, checkpoints), where seeds is the number of seeds,
                              and checkpints is the number of checkpoints.
        """
        seeds = DataReader.find_option_values('seed', experiment)
        result = {}
        steps = []
        min_length = 100
        max_length = 0

        for seed in seeds:
            result[seed] = []
            checkpoints = DataReader.find_option_values(
                'checkpoint', experiment, seed)
            if len(steps) < len(checkpoints):
                steps = checkpoints

            for checkpoint in checkpoints:
                success = DataReader.get_success_rate(
                    experiment, seed, checkpoint)
                result[seed].append(success)

            min_length = min(min_length, len(result[seed]))
            max_length = max(max_length, len(result[seed]))

        if len(result) > 0:
            result = np.stack([np.pad(np.array(result[seed]), (0, max_length - len(result[seed])), 'edge')
                               for seed in result])
            steps = np.array(steps)
            return steps, result
        else:
            return None, None

    @staticmethod
    def get_learning_curves_for_seed(experiment, seed):
        """Gets the training and validation total losses for a given experiment
        and seed.
        """
        path = DataReader.get_training_log_file(experiment, seed)
        with open(path, 'r') as f:
            lines = f.readlines()
        regex = re.compile(".*step\s(\d+).*\s\[.*\π\:\s(.*)\].*\[.*\π\:\s(.*)\]")
        steps = []
        train_losses = []
        validation_losses = []
        for line in lines:
            match = regex.match(line)
            if match:
                steps.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                validation_losses.append(float(match.group(3)))
        result = dict(
            steps=steps,
            train_losses=train_losses,
            validation_losses=validation_losses,
        )
        return result

    @staticmethod
    def get_learning_curves_for_experiment(experiment):
        seeds = DataReader.find_option_values('seed', experiment)
        result = {}
        steps = []
        min_length = 100
        max_length = 0

        train = {}
        validation = {}

        for seed in seeds:
            result[seed] = []
            curves = DataReader.get_learning_curves_for_seed(experiment, seed)
            for i, step in enumerate(curves['steps']):
                train.setdefault(step, []).append(curves['train_losses'][i])
                validation.setdefault(step, []).append(curves['validation_losses'][i])

        train_means = []
        train_stds = []
        validation_means = []
        validation_stds = []

        for key in train:
            train_means.append(float(np.mean(train[key])))
            train_stds.append(float(np.std(train[key])))
            validation_means.append(float(np.mean(validation[key])))
            validation_stds.append(float(np.std(validation[key])))

        result = dict(
            steps=list(train.keys()),
            train=(train_means, train_stds),
            validation=(validation_means, validation_stds),
        )
        return result


    @staticmethod
    def get_episodes_with_outcome(experiment, seed, step, outcome):
        """Gets episodes with given outcome for a given model.
        If outcome == 1, returns successful episodes,
        if outcome == 0, returns failing episodes.
        """
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
    def get_episode_success_map(experiment, seed, step):
        """Gets a 0-1 array of shape (episodes) where episodes is
        the number of episodes.

        Ith value in the result is 0 if the ith episode failed,
        and 1 otherwise.
        """
        successes = DataReader.get_episodes_with_outcome(experiment,
                                                         seed,
                                                         step,
                                                         1)
        successes = np.array(successes) - 1
        result = np.zeros(EPISODES)
        result[successes] = 1
        return result

    @staticmethod
    def get_episodes_success_counts(experiment):
        """For a given experiment, for all episodes checks performance of all
        the models with all possible seeds and checkpoints, and returns
        an array of shape (episodes) where episodes is the number of episodes,
        where Ith value is the number of models in this experiment that
        succeeded in this episode.
        """
        seeds = DataReader.find_option_values('seed', experiment)
        result = np.zeros(EPISODES)
        for seed in seeds:
            checkpoints = DataReader.find_option_values(
                'checkpoint', experiment, seed)
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
        """ Returns an array of speeds for given model and given episode"""
        return DataReader.get_model_speeds(experiment,
                                           seed,
                                           checkpoint)[episode - 1]

    @staticmethod
    def get_episode_costs(experiment, seed, checkpoint, episode):
        """ Returns an array of data frames with all the costs for 
        given evaluation """
        costs = DataReader.get_model_costs(experiment,
                                           seed,
                                           checkpoint)
        if costs is not None:
            return costs[episode - 1]
        else:
            return None

    @staticmethod
    @lru_cache(maxsize=10)
    def get_model_costs(experiment, seed, checkpoint):
        """ Returns an array of costs for given model for all episodes"""
        path = DataReader.get_experiments_mapping()[experiment]
        regex = path[0] + 'planning_results/' + path[1] + \
            f'-seed={seed}-novaluestep{checkpoint}' + '.model.costs'
        costs_paths = glob(regex)
        if len(costs_paths) == 0:
            print(
                f'costs_paths for {regex} is {costs_paths} and it\'s length is not 1')
            return None
        else:
            raw_costs = torch.load(costs_paths[0])
            # list of DataFrame, one per episode
            costs = [pandas.DataFrame(cost if type(cost) == type([]) else cost.tolist()) for cost in raw_costs]
            return costs

    @staticmethod
    @lru_cache(maxsize=10)
    def get_model_speeds(experiment, seed, checkpoint):
        """ Returns an array of speeds for given model for all episodes"""
        path = DataReader.get_experiments_mapping()[experiment]
        regex = path[0] + 'planning_results/' + path[1] + \
            f'-seed={seed}-novaluestep{checkpoint}' + '.model.states'
        states_paths = glob(regex)
        assert len(states_paths) == 1, \
            f'states_paths for {regex} is {states_paths} and it\'s length is not 1'
        states_path = states_paths[0]
        states = torch.load(states_path)

        result = []
        for i in range(len(states)):
            episode_states = states[i]
            episode_states = list(map(lambda x: x[-1], episode_states))
            episode_states = torch.stack(episode_states)
            result.append(episode_states[:, 2:].norm(dim=1))  # is it correct
        return result

    @staticmethod
    @lru_cache(maxsize=10)
    def get_model_states(experiment, seed, checkpoint):
        """ Returns an array of states for given model for all episodes"""
        path = DataReader.get_experiments_mapping()[experiment]
        regex = path[0] + 'planning_results/' + path[1] + \
            f'-seed={seed}-novaluestep{checkpoint}' + '.model.states'
        states_paths = glob(regex)
        assert len(states_paths) == 1, \
            f'states_paths for {regex} is {states_paths} and it\'s length is not 1'
        states_path = states_paths[0]
        states = torch.load(states_path)

        result = []
        for i in range(len(states)):
            episode_states = states[i]
            episode_states = list(map(lambda x: x[-1], episode_states))
            episode_states = torch.stack(episode_states)
            result.append(episode_states)
        return result
