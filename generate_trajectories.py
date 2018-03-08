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
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
opt = parser.parse_args()

opt.state_image = (opt.state_image == 1)

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

os.system("mkdir -p " + opt.data_dir)

data_file = f'{opt.data_dir}/traffic_data_lanes={opt.lanes}-episodes={opt.n_episodes}-seed={opt.seed}.pkl'
print(f'Will save as {data_file}')

register(
    id='Traffic-v0',
    entry_point='traffic_gym:StatefulEnv',
    tags={'wrapper_config.TimeLimit.max_episodesteps': 100},
    kwargs={'display': opt.display,
            'nb_lanes': opt.lanes,
            'traffic_rate': opt.traffic_rate,
            'state_image': opt.state_image},
)

env = gym.make('Traffic-v0')

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


def run_episode(ep):
    action = np.array([0, 0, 1, 0, 0, 0])
    states_, actions_, rewards_ = [], [], []
    done = False

    state, objects = env.reset()
    for t in range(500):
        try:
            state, reward, done, vehicles = env.step(None)
            env.render()
        except:
            print('exception, breaking')
            break

        if env.collision:
            print('collision, breaking')
            break

    runs = []

    '''
    vid = 0
    for v in vehicles:
        im = v._states_image[100:]
        save_dir = 'videos/states/ex{:d}'.format(vid)
        os.system('mkdir -p ' + save_dir)
        for t in range(len(im)):
            scipy.misc.imsave('{}/im{:05d}.png'.format(save_dir, t), im[t])
        vid += 1
    '''

    for v in vehicles:
        images = torch.stack(v._states_image).permute(0, 3, 2, 1)
        states, masks, actions = prepare_trajectory_state(v._states, v._actions)
        assert(states.size(0) == images.size(0))
        # remove the first part, so cars don't appear from outside the frame
        if images.size(0) > 100:
            images = images[100:]
            states = states[100:]
            masks = masks[100:]
            actions = actions[100:]
            runs.append({'states': states, 'masks': masks, 'actions': actions, 'images': images})

    return runs


episodes = []
for i in range(opt.n_episodes):
    print(f'episode {i + 1}/{opt.n_episodes}')
    runs = run_episode(i)
    episodes += runs

pickle.dump(episodes, open(data_file, 'wb'))
