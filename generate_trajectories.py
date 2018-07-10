import argparse, pdb
import gym
import numpy as np
import os
import pickle
import random
import torch
import scipy.misc
from gym.envs.registration import register

parser = argparse.ArgumentParser()
parser.add_argument('-display', type=int, default=0)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-lanes', type=int, default=3)
parser.add_argument('-traffic_rate', type=int, default=15)
parser.add_argument('-n_episodes', type=int, default=1000)
parser.add_argument('-state_image', type=int, default=1)
parser.add_argument('-save_images', type=int, default=0)
parser.add_argument('-store', type=int, default=1)
parser.add_argument('-data_dir', type=str, default='scratch/nvidia-collab/data/')
parser.add_argument('-steps', type=int, default=500)
parser.add_argument('-fps', type=int, default=30)
parser.add_argument('-time_slot', type=int, default=0)
parser.add_argument('-map', type=str, default='i80', choices={'ai', 'i80', 'us101'})
opt = parser.parse_args()

opt.state_image = (opt.state_image == 1)
opt.store = (opt.store == 1)

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

os.system("mkdir -p " + opt.data_dir)

kwargs = {
    'display': opt.display,
    'state_image': opt.state_image,
    'store': opt.store,
    'fps': opt.fps,
    'nb_lanes': opt.lanes,
    'traffic_rate': opt.traffic_rate,
}

register(
    id='Traffic-v0',
    entry_point='traffic_gym:StatefulEnv',
    kwargs=kwargs
)

register(
    id='Traffic-v1',
    entry_point='traffic_gym_v1:RealTraffic',
    kwargs=kwargs
)

gym.envs.registration.register(
    id='US-101-v0',
    entry_point='map_us101:US101',
    kwargs=kwargs,
)

env_names = {
    'ai': 'Traffic-v0',
    'i80': 'Traffic-v1',
    'us101': 'US-101-v0',
}

print('Building the environment (loading data, if any)')
env = gym.make(env_names[opt.map])


def run_episode():
    env.reset(frame=0, time_slot=opt.time_slot)
    while True:
        env.step(None)
        env.render()


episodes = []
for i in range(opt.n_episodes):
    run_episode()
    episodes += runs
