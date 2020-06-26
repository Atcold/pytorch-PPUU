"""A class with static methods which can be used to access the data about
experiments.
This includes reading logs to parse success cases, reading images, costs
and speed.
"""

from glob import glob
from io import BytesIO
import re
import json
import os
from functools import lru_cache

import torch
import pandas
import numpy as np
import imageio
from torchvision.transforms import ToPILImage


EPISODES = 561


class DataReader:
    """Container class for the static data access methods"""

    # EXPERIMENTS_MAPPING_FILE = "experiments_mapping.json"

    # @staticmethod
    # @lru_cache(maxsize=1)
    # def get_experiments_mapping():
    #     """Reads the experiments mapping from a json file
    #     EXPERIMENTS_MAPPING_FILE
    #     """
    #     with open(DataReader.EXPERIMENTS_MAPPING_FILE, "r") as f:
    #         x = json.load(f)
    #     return x

    @staticmethod
    def tensor_to_image(tensor):
        image = ToPILImage()(tensor)
        b = BytesIO()
        image.save(b, format="png")
        return b.getvalue()

    @staticmethod
    def get_images(experiment, version, checkpoint, episode):
        """Get simulator images for a given model evaluation on a
        given episode"""
        images = (
            DataReader.get_episode_result(
                experiment, version, checkpoint, episode
            )["images"]
            .detach()
            .cpu()
        )
        result = []
        for i in range(images.shape[0]):
            result.append(DataReader.tensor_to_image(images[i]))
        return result

    @staticmethod
    def get_gradients(experiment, version, checkpoint, episode):
        """Get gradients for a given model evaluation on a given episode"""
        gradients = (
            DataReader.get_episode_result(
                experiment, version, checkpoint, episode
            )["gradients"]
            .detach()
            .cpu()
        )
        if len(gradients.shape) == 5:
            gradients = gradients[0]
        images = []
        for i in range(gradients.shape[0]):
            images.append(DataReader.tensor_to_image(gradients[i]))
        return images

    @staticmethod
    def get_last_gradient(experiment, seed, checkpoint, episode):
        """Get the last gradient for the model and episode

        Returns:
            (value, x, y) - tuple, where value is the max value of the
                            gradient, x, y are the location of this max
                            value in the  gradient image.
        """
        image = (
            DataReader.get_episode_result(
                experiment, version, checkpoint, episode
            )["gradients"]
            .detach()
            .cpu()
        )
        if len(image.shape) == 5:
            image = image[0]
        image = image[-1]
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
        regex = (
            path[0]
            + "planning_results/"
            + path[1]
            + f"-seed={seed}-novaluestep{step}"
            + ".model.log"
        )
        paths = glob(regex)
        assert (
            len(paths) == 1
        ), f"paths for {regex} is not length of 1, and is equal to {paths}"
        return paths[0]

    @staticmethod
    def get_experiments_path():
        return "/home/us441/nvidia-collab/vlad/results/refactored_test"

    @staticmethod
    def get_experiment_path(experiment):
        return os.path.join(DataReader.get_experiments_path(), experiment)

    @staticmethod
    def get_version_checkpoints_path(experiment, version):
        return os.path.join(
            DataReader.get_experiment_path(experiment),
            version,
            "evaluation_results",
        )

    @staticmethod
    def get_episodes_path(experiment, version, checkpoint):
        return os.path.join(
            DataReader.get_version_checkpoints_path(experiment, version),
            checkpoint,
            "episode_data",
        )

    @staticmethod
    def get_episode_result_path(experiment, version, checkpoint, episode):
        return os.path.join(
            DataReader.get_episodes_path(experiment, version, checkpoint),
            str(episode),
        )

    @staticmethod
    def get_episode_result(experiment, version, checkpoint, episode):
        path = DataReader.get_episode_result_path(
            experiment, version, checkpoint, episode
        )
        return torch.load(path)

    @staticmethod
    def get_evaluation_result_path(experiment, version, checkpoint):
        return os.path.join(
            DataReader.get_version_checkpoints_path(experiment, version),
            checkpoint,
            "evaluation_results.json",
        )

    @staticmethod
    def get_evaluation_result(experiment, version, checkpoint):
        path = DataReader.get_evaluation_result_path(
            experiment, version, checkpoint
        )
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    @lru_cache(maxsize=100)
    def find_experiments():
        experiments_path = DataReader.get_experiments_path()
        experiments = os.listdir(experiments_path)
        return sorted(experiments)

    @staticmethod
    @lru_cache(maxsize=100)
    def find_experiment_versions(experiment):
        experiment_path = DataReader.get_experiment_path(experiment)
        versions = os.listdir(experiment_path)
        return sorted(versions)

    @staticmethod
    @lru_cache(maxsize=100)
    def find_version_checkpoints(experiment, version):
        version_checkpoints_path = DataReader.get_version_checkpoints_path(
            experiment, version
        )
        return sorted(os.listdir(version_checkpoints_path))

    @staticmethod
    @lru_cache(maxsize=100)
    def find_checkpoint_episodes(experiment, version, checkpoint):
        episodes_results_path = DataReader.get_episodes_path(
            experiment, version, checkpoint
        )
        return sorted(os.listdir(episodes_results_path))

    def find_option_values(
        option, experiment=None, seed=None, checkpoint=None
    ):
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
        if option == "seed":
            path = DataReader.get_experiments_mapping()[experiment]
            logs = glob(path[0] + "planning_results/" + path[1] + "*.log")
            regexp = r"seed=(\d+)-"
        elif option == "checkpoint":
            path = DataReader.get_experiments_mapping()[experiment]
            logs = glob(
                path[0]
                + "planning_results/"
                + path[1]
                + f"-seed={seed}"
                + "*.model.log"
            )
            regexp = r"-novaluestep(\d+)\."
        elif option == "episode":
            path = DataReader.get_experiments_mapping()[experiment]
            logs = glob(
                path[0]
                + "planning_results/videos_simulator/"
                + path[1]
                + f"-seed={seed}-novaluestep{checkpoint}.model/ep*"
            )
            regexp = r"model/ep(\d+)"

        values = []

        for log in logs:
            m = re.search(regexp, log)
            if m:
                result = m.group(1)
                values.append(int(result))
            else:
                print(f"{log} doesn't contain {option}")

        # log files for each step are generated for seeds
        values = list(set(values))
        values.sort()

        return values

    @staticmethod
    def get_success_rate(experiment, version, checkpoint):
        """get the success rate for a given model"""
        evaluation_result = DataReader.get_evaluation_result(
            experiment, version, checkpoint
        )
        return evaluation_result["stats"]["success_rate"]

    @staticmethod
    def get_version_success_rates(experiment, version):
        checkpoints = DataReader.find_version_checkpoints(experiment, version)
        result = {}
        for checkpoint in checkpoints:
            checkpoint_int = int(re.findall(r"\d+", checkpoint)[0])
            result[checkpoint_int] = DataReader.get_success_rate(
                experiment, version, checkpoint
            )
        return result

    @staticmethod
    def get_success_rates_for_experiment(experiment):
        """get success rate arrays for each seed for the given experiment
        across all checkpoints.
        The resulting shape of the np array is
        (versions, checkpoints), where versions is the number of versions,
                              and checkpoints is the number of checkpoints.
        """
        results = {}
        for version in DataReader.find_experiment_versions(experiment):
            success_rates = DataReader.get_version_success_rates(
                experiment, version
            )
            for checkpoint, success_rate in success_rates.items():
                results.setdefault(checkpoint, []).append(success_rate)

        keys = sorted(results.keys())
        means = [float(np.mean(results[x])) for x in keys]
        stds = [float(np.std(results[x])) for x in keys]
        result = dict(checkpoints=keys, means=means, stds=stds,)
        return result

    @staticmethod
    def get_episodes_with_outcome(experiment, seed, step, outcome):
        """Gets episodes with given outcome for a given model.
        If outcome == 1, returns successful episodes,
        if outcome == 0, returns failing episodes.
        """
        path = DataReader.get_evaluation_log_file(experiment, seed, step)
        with open(path, "r") as f:
            lines = f.readlines()
        regex = re.compile(r".*ep:\s+(\d+).*\|\ssuccess:\s+(\d).*")
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
        successes = DataReader.get_episodes_with_outcome(
            experiment, seed, step, 1
        )
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
        seeds = DataReader.find_option_values("seed", experiment)
        result = np.zeros(EPISODES)
        for seed in seeds:
            checkpoints = DataReader.find_option_values(
                "checkpoint", experiment, seed
            )
            for checkpoint in checkpoints:
                success = DataReader.get_episodes_with_outcome(
                    experiment, seed, checkpoint, 1
                )
                success = np.array(success)
                success = success - 1
                one_hot = np.zeros((len(success), EPISODES))
                one_hot[np.arange(len(success)), success] = 1
                one_hot = (np.sum(one_hot, axis=0),)
                one_hot = np.squeeze(one_hot)
                result += one_hot
        return result

    @staticmethod
    def get_episode_speeds(experiment, seed, checkpoint, episode):
        """ Returns an array of speeds for given model and given episode"""
        return DataReader.get_model_speeds(experiment, seed, checkpoint)[
            episode - 1
        ]

    @staticmethod
    def get_episode_costs(experiment, seed, checkpoint, episode):
        """ Returns an array of data frames with all the costs for
        given evaluation """
        costs = DataReader.get_model_costs(experiment, seed, checkpoint)
        if costs is not None:
            return costs[episode - 1]
        else:
            return None

    @staticmethod
    @lru_cache(maxsize=10)
    def get_model_costs(experiment, seed, checkpoint):
        """ Returns an array of costs for given model for all episodes"""
        return DataReader.get_evaluation_result(experiment, seed, checkpoint)[
            "cost_sequence"
        ]
        # raw_costs = torch.load(costs_paths[0])
        # # list of DataFrame, one per episode
        # costs = [
        #     pandas.DataFrame(
        #         cost if type(cost) == type([]) else cost.tolist()
        #     )
        #     for cost in raw_costs
        # ]
        # return costs

    @staticmethod
    @lru_cache(maxsize=10)
    def get_model_speeds(experiment, seed, checkpoint):
        """ Returns an array of speeds for given model for all episodes"""
        states = DataReader.get_evaluation_result(
            experiment, seed, checkpoint
        )["state_sequence"]
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
        states = DataReader.get_evaluation_result(
            experiment, seed, checkpoint
        )["state_sequence"]
        result = []
        for i in range(len(states)):
            episode_states = states[i]
            episode_states = list(map(lambda x: x[-1], episode_states))
            episode_states = torch.stack(episode_states)
            result.append(episode_states)
        return result
