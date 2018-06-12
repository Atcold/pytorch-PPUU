import argparse
import random
import torch
import numpy
import gym

parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-nb_conditions', type=int, default=10)
parser.add_argument('-nb_predictions', type=int, default=10)
parser.add_argument('-nb_samples', type=int, default=1)
parser.add_argument('-models_dir', type=str, default='./models_il/')
parser.add_argument('-display', type=int, default=0)
parser.add_argument('-v', type=str, default='3', choices={'3'})

opt = parser.parse_args()

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)

kwargs = {
    'fps': 50,
    'nb_states': opt.nb_conditions,
    'display': opt.display
}

gym.envs.registration.register(
    id='Traffic-v3',
    entry_point='traffic_gym_v3:ControlledI80',
    kwargs=kwargs,
)

print('Building the environment (loading data, if any)')
env = gym.make('Traffic-v' + opt.v)

for episode in range(10):
    env.reset()
    done = False
    while not done:
        observation, reward, done, info = env.step(numpy.zeros((2,)))
        # print(observation, reward, done, info)
        env.render()

print('Done')
