import argparse, pdb
import gym
import numpy as np
import os
import pickle
import random
import torch
from torch.autograd import Variable
from gym.envs.registration import register

parser = argparse.ArgumentParser()
parser.add_argument('-display', type=int, default=1)
parser.add_argument('-seed', type=int, default=9999)
parser.add_argument('-lanes', type=int, default=3)
parser.add_argument('-traffic_rate', type=int, default=15)
parser.add_argument('-n_episodes', type=int, default=10)
parser.add_argument('-ncond', type=int, default=4)
opt = parser.parse_args()

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

register(
    id='Traffic-v0',
    entry_point='traffic_gym:StatefulEnv',
    tags={'wrapper_config.TimeLimit.max_episodesteps': 100},
    kwargs={'display': opt.display,
            'nb_lanes': opt.lanes,
            'store': True,
            'traffic_rate': opt.traffic_rate},
)

env = gym.make('Traffic-v0')
policy = torch.load('models/model=policy-cnn-ncond=4-npred=50-lrt=0.0001-nhidden=100-nfeature=64.model')

# parse out the mask and state. This is specific to using the (x, y, dx, dy) state,
# not used for images.
def prepare_trajectory(states, actions):
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
    action = np.array([0, 0, 1, 0, 0, 0])
    states_, actions_, rewards_ = [], [], []
    done = False

    state, objects = env.reset()
    for t in range(2000):
        if t > 20:
            v = None
            for v_ in vehicles:
                if v_._id == env.policy_car_id:
                    v = v_

            images = torch.stack(v._states_image).permute(0, 3, 2, 1).float()
            images /= 255.0
            images = Variable(images[-opt.ncond:].clone().float().unsqueeze(0))
            states, masks, actions = prepare_trajectory(v._states, v._actions)
            states = Variable(states[-opt.ncond:, 0].clone().unsqueeze(0))
            masks = Variable(masks[-opt.ncond:].unsqueeze(0))
            action, _ = policy(images, states, actions)
            print(f'dv = {action.data[0][0][2]:0.4f}, (dx, dy) = ({action.data[0][0][0]:0.4f}, {action.data[0][0][1]:0.4f})')
            action = action.data[0][0].numpy()
        else:
            action = None
        state, reward, done, vehicles = env.step(action)
        env.render()


for i in range(opt.n_episodes):
    print(f'episode {i + 1}/{opt.n_episodes}')
    runs = run_episode()
