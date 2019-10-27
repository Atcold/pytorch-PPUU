import sklearn
import sklearn.decomposition
import numpy as np

from DataReader import DataReader


class PCA:

    def __init__(self):
        features = PCA.get_pca_data()
        self.pca = sklearn.decomposition.PCA(n_components=2)
        self.pca.fit(features)

    def transform(self, value):
        return self.pca.transform(value)

    @staticmethod
    def get_episode_features(experiment, seed, checkpoint, episode):
        history_size = 10
        features = []
        features.append(DataReader.get_episode_speeds(experiment, seed, checkpoint, episode)[-history_size:])
        costs = DataReader.get_episode_costs(experiment, seed, checkpoint, episode)
        columns_to_save = ['proximity_cost', 'lane_cost', 'pixel_proximity_cost']
        for column in columns_to_save:
            features.append(costs[column].to_numpy()[-history_size:])
        features = np.stack(features)
        features = features.flatten()
        return features

    @staticmethod
    def get_pca_data():
        # we don't have costs for all values, should do that.
        experiment = 'Deterministic policy, regressed cost'
        seed = 3
        checkpoint = 25000
        failing = DataReader.get_episodes_with_outcome(experiment, seed, checkpoint, 0)
        data = []
        for fail in failing[:4]:
            print(fail)
            features = PCA.get_episode_features(experiment, seed, checkpoint, fail)
            data.append(features)

        data = np.stack(data)
        print(data.shape)
        return data
