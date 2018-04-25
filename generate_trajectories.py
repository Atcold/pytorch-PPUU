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
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-display', type=int, default=0)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-lanes', type=int, default=3)
parser.add_argument('-traffic_rate', type=int, default=15)
parser.add_argument('-n_episodes', type=int, default=1000)
parser.add_argument('-state_image', type=int, default=1)
parser.add_argument('-save_images', type=int, default=0)
parser.add_argument('-store', type=int, default=1)
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
parser.add_argument('-steps', type=int, default=500)
parser.add_argument('-v', type=str, default='0')
parser.add_argument('-fps', type=int, default=30)
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
    }

#data_file = f'{opt.data_dir}/traffic_data_lanes={opt.lanes}-episodes={opt.n_episodes}-seed={opt.seed}.pkl'

if opt.dataset == 'simulator':
 #   print(f'will save as {data_file}')
    register(
        id='Traffic-v0',
        entry_point='traffic_gym:StatefulEnv',
        kwargs=kwargs
    )
    kwargs = {
        'display': opt.display,
        'nb_lanes': opt.lanes,
        'traffic_rate': opt.traffic_rate,
        'state_image': opt.state_image,
        'store': opt.store,
    }

elif opt.dataset == 'i80':
    opt.steps = 10000000000

    register(
        id='Traffic-v1',
        entry_point='traffic_gym_v1:RealTraffic',
        kwargs=kwargs
    )

print('Building the environment (loading data, if any)')
env = gym.make('Traffic-v' + opt.v)


# parse out the mask and state. This is specific to using the (x, y, dx, dy) state,
# not used for images.
def prepare_trajectory_state(states, actions):
    # parse out the masks
    T = len(states)
    s, m = [], []
    for t in range(T):
        s.append(states[t][0])
        m.append(states[t][1])

    s = torch.stack(s)
    m = torch.stack(m)
    a = torch.stack(actions)
    return s, m, a


def run_episode():
    env.reset()
    for t in range(opt.steps):
        state, reward, vehicles = env.step(None)
        env.render()

        if env.collision:
            print('collision, breaking')
            break
    runs = []
    vehicles = env.vehicles
    if opt.save_images == 1:
        for v in vehicles:
#            save_dir = f'videos/states/ex{vid:d}'
            v.dump_state_image(save_dir)

    for v in vehicles:
        if len(v._states_image) > 1:
            images = torch.stack(v._states_image).permute(0, 3, 2, 1)
            states, masks, actions = prepare_trajectory_state(v._states, v._actions)
            runs.append({'states': states, 'masks': masks, 'actions': actions, 'images': images})

    return runs


episodes = []
for i in range(opt.n_episodes):
#    print(f'[episode {i + 1}]')
    runs = run_episode()
    episodes += runs

pickle.dump(episodes, open(data_file, 'wb'))
