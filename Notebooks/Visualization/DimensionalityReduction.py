"""Contains a class used for clustering and dimensionality reduction
of episode features for failure analysis."""
import sklearn
import sklearn.decomposition
import sklearn.manifold
import sklearn.preprocessing
import sklearn.cluster
import numpy as np
import pickle
import os
import traceback

from DataReader import DataReader


class DimensionalityReduction:
    """Dimensionality reduction and clustering class"""

    SAVE_DIMENSIONALITY_REDUCTION_PATH = './dimensionality_reduction.pickle'
    """ Path for saving dimensionality reduction class state for
    faster loading"""
    SAVE_KM_PATH = './km.pickle'
    """ Path for saving clustering class state for faster loading"""

    def __init__(self):
        """Lazy initialization, in order to not waste time every time we open the notebook"""
        self.initialized = False

    def initialize(self):
        """If no saved state data found, loads the data for dimensionality
        reduction and fit the dimensionality reduction and clustering
        algorithms on the data. Fitting the data may take time.
        Otherwise loads the saved state data.
        """
        if self.initialized:
            return
        if os.path.exists(DimensionalityReduction.SAVE_DIMENSIONALITY_REDUCTION_PATH):
            with open(DimensionalityReduction.SAVE_DIMENSIONALITY_REDUCTION_PATH, 'rb') as f:
                self.dimensionality_reduction = pickle.load(f)
            with open(DimensionalityReduction.SAVE_KM_PATH, 'rb') as f:
                self.km = pickle.load(f)
            print('loaded from pickle')
        else:
            print('creating dimensionality_reduction')
            features = DimensionalityReduction.get_dimensionality_reduction_data()
            # features = np.random.rand(100, 50)
            # self.dimensionality_reduction = sklearn.decomposition.PCA(n_components=2)
            self.dimensionality_reduction = sklearn.manifold.TSNE(
                n_components=2)
            self.dimensionality_reduction.fit(features)
            self.km = sklearn.cluster.KMeans(n_clusters=5).fit(features)
            with open(DimensionalityReduction.SAVE_DIMENSIONALITY_REDUCTION_PATH, 'wb') as f:
                pickle.dump(self.dimensionality_reduction, f)
            with open(DimensionalityReduction.SAVE_KM_PATH, 'wb') as f:
                pickle.dump(self.km, f)
        self.initialized = True

    def transform(self, value):
        """ Performs dimensionality reduction on given values"""
        self.initialize()
        return self.dimensionality_reduction.fit_transform(value)

    def cluster(self, value):
        """ Performs clustering on given values"""
        self.initialize()
        return self.km.predict(value)

    @staticmethod
    def get_episode_features(experiment, seed, checkpoint, episode):
        """ Get features for one episode
        This is used for dimensionality reduction, which is later used for
        scatter plotting.
        """
        history_size = 10
        features = []
        features.append(DataReader.get_episode_speeds(
            experiment, seed, checkpoint, episode)[-history_size:])
        costs = DataReader.get_episode_costs(
            experiment, seed, checkpoint, episode)
        columns_to_save = ['proximity_cost',
                           'lane_cost', 'pixel_proximity_cost']
        for column in columns_to_save:
            features.append(costs[column].to_numpy()[-history_size:])
        features = np.stack(features)
        features = features.flatten()
        return features

    @staticmethod
    def get_model_failing_features(experiment, seed, checkpoint):
        """ Get features for one model 
        This is used for dimensionality reduction, which is later used for
        scatter plotting.
        """
        speeds = DataReader.get_model_speeds(experiment, seed, checkpoint)
        costs = DataReader.get_model_costs(experiment, seed, checkpoint)
        states = DataReader.get_model_states(experiment, seed, checkpoint)
        failing = DataReader.get_episodes_with_outcome(
            experiment, seed, checkpoint, 0)
        #failing = DataReader.find_option_values('episode', experiment, seed, checkpoint)

        data = []

        history_size = 10

        for fail in failing:
            # features = DimensionalityReduction.get_episode_features(experiment, seed, checkpoint, fail)
            columns_to_save = ['lane_cost', 'pixel_proximity_cost']
            features = [speeds[fail - 1][-history_size:]]
            for c in columns_to_save:
                features.append(costs[fail - 1][c][-history_size:])
            for i in range(len(features)):
                features[i] = np.pad(
                    features[i], (history_size - features[i].shape[0], 0), 'constant')
            features = np.stack(features)
            features = features.flatten()
            l = costs[fail - 1]['collisions_per_frame'].shape[0]
            features = np.append(
                features, [costs[fail - 1]['collisions_per_frame'][l - 1]])
            features = np.append(
                features, [costs[fail - 1]['arrived_to_dst'][l - 1]])
            features = np.append(features, [states[fail - 1][l - 1][0]])
            features = np.append(features, [states[fail - 1][l - 1][1]])
            features = np.append(features, DataReader.get_last_gradient(
                experiment, seed, checkpoint, fail))
            data.append(features)

        data = np.stack(data)
        data = sklearn.preprocessing.scale(data)

        return data

    @staticmethod
    def get_dimensionality_reduction_data():
        """ Get features for all models
        This is used for dimensionality reduction, which is later used for
        scatter plotting.
        """
        # we don't have costs for all values, should do that.
        data = []
        experiment = 'Deterministic policy, regressed cost'
        seed = 3
        checkpoint = 25000
        seeds = DataReader.find_option_values('seed', experiment)
        for seed in seeds:
            checkpoints = DataReader.find_option_values(
                'checkpoint', experiment, seed)
            for checkpoint in checkpoints:
                try:
                    if data == []:
                        data = DimensionalityReduction.get_model_failing_features(
                            experiment, seed, checkpoint)
                    else:
                        data = np.concatenate([data, DimensionalityReduction.get_model_failing_features(
                            experiment, seed, checkpoint)])
                except Exception as e:
                    print(checkpoint, 'failed', e)
                    traceback.print_exc()
        data = sklearn.preprocessing.scale(data)
        return data
