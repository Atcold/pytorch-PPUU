import argparse
import random
import torch
import numpy
import gym

parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-map', type=str, default='i80')
parser.add_argument('-nb_conditions', type=int, default=10)
parser.add_argument('-nb_predictions', type=int, default=10)
parser.add_argument('-nb_samples', type=int, default=1)
parser.add_argument('-models_dir', type=str, default='./models_il/')
parser.add_argument('-v', type=str, default='3', choices={'3'})

opt = parser.parse_args()

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)

kwargs = {
    'fps': 50,
    'nb_states': opt.nb_conditions,
}


gym.envs.registration.register(
    id='I-80-v1',
    entry_point='map_i80_ctrl:ControlledI80',
    kwargs={'fps': 10, 'nb_states': 20, 'display': 0, 'delta_t': 0.1},
)

print('Building the environment (loading data, if any)')
env_names = {
    'i80': 'I-80-v1',
}

env = gym.make(env_names[opt.map])


for episode in range(10):

    env.reset()

    done = False
    while not done:
        observation, reward, done, info = env.step(numpy.zeros((2,)))
        print(reward)
#        print(observation, reward, done, info)
        env.render()
    print('end of episode')

print('Done')
