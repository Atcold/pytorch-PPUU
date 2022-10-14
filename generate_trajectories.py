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
parser.add_argument('-state_image', type=int, default=1)
parser.add_argument('-save_images', type=int, default=0)
parser.add_argument('-store', type=int, default=1)
parser.add_argument('-data_dir', type=str, default='traffic-data/state-action-cost/')
parser.add_argument('-fps', type=int, default=30)
parser.add_argument('-time_slot', type=int, default=0)
parser.add_argument('-map', type=str, default='i80', choices={'ai', 'i80', 'us101', 'lanker', 'peach'})
parser.add_argument('-delta_t', type=float, default=0.1)
opt = parser.parse_args()

opt.state_image = (opt.state_image == 1)
opt.store = (opt.store == 1)

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

os.system("mkdir -p " + opt.data_dir)

kwargs = dict(
    display=opt.display,
    state_image=opt.state_image,
    store=opt.store,
    fps=opt.fps,
    nb_lanes=opt.lanes,
    traffic_rate=opt.traffic_rate,
    data_dir=opt.data_dir,
    delta_t=opt.delta_t,
)

register(
    id='Traffic-v0',
    entry_point='traffic_gym:Simulator',
    kwargs=kwargs
)

register(
    id='I-80-v0',
    entry_point='map_i80:I80',
    kwargs=kwargs
)

gym.envs.registration.register(
    id='US-101-v0',
    entry_point='map_us101:US101',
    kwargs=kwargs,
)

gym.envs.registration.register(
    id='Lankershim-v0',
    entry_point='map_lanker:Lankershim',
    kwargs=kwargs,
)

gym.envs.registration.register(
    id='Peachtree-v0',
    entry_point='map_peach:Peachtree',
    kwargs=kwargs,
)

env_names = {
    'ai': 'Traffic-v0',
    'i80': 'I-80-v0',
    'us101': 'US-101-v0',
    'lanker': 'Lankershim-v0',
    'peach': 'Peachtree-v0',
}

print('Building the environment (loading data, if any)')
env = gym.make(env_names[opt.map])

env.reset(frame=0, time_slot=opt.time_slot)
done = False
while not done:
    observation, reward, done, info = env.step(np.zeros((2,)))
    env.render()

print(f'Data generation for <{opt.map}, time slot {opt.time_slot}> completed')
