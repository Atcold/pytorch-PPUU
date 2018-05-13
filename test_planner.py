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
parser.add_argument('-v', type=str, default='2', choices={'2'})

opt = parser.parse_args()

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)

kwargs = {
    'fps': 50,
}

gym.envs.registration.register(
    id='Traffic-v2',
    entry_point='traffic_gym_v2:RealTraffic',
    kwargs=kwargs,
)

print('Building the environment (loading data, if any)')
env = gym.make('Traffic-v' + opt.v)

env.reset(frame=2000, control={
    'lane': 3,
    'nb_states': opt.nb_conditions,
})

done = False
while not done:
    observation, reward, done, info = env.step(numpy.zeros((2,)))
    env.render()

print('Done')
