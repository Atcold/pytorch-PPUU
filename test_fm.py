import argparse, pdb
import gym
import numpy as np
import os
import pickle
import random
import torch
from torch.autograd import Variable
from gym.envs.registration import register
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument('-display', type=int, default=1)
parser.add_argument('-seed', type=int, default=9999)
parser.add_argument('-lanes', type=int, default=3)
parser.add_argument('-traffic_rate', type=int, default=15)
parser.add_argument('-n_episodes', type=int, default=10)
parser.add_argument('-ncond', type=int, default=4)
parser.add_argument('-npred', type=int, default=10)
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
            'traffic_rate': opt.traffic_rate},
)

env = gym.make('Traffic-v0')
#model = torch.load('models/model=fwd-cnn-ncond=4-npred=10-lrt=0.001-nhidden=100-nfeature=32.model')
model = torch.load('models/model=fwd-cnn-ncond=4-npred=10-lrt=0.0001-nhidden=100-nfeature=64.model')

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
        if t > 300:
            vid = 0
            for v in vehicles:
                images = torch.stack(v._states_image).permute(0, 3, 2, 1).float()
                images /= 255.0
                images = Variable(images[-opt.ncond:].clone().float().unsqueeze(0))
                actions = torch.zeros(1, opt.npred, 3)
                actions[:, :, 2].fill_(0)
                actions[:, :, 1].fill_(+0.3)
                actions = Variable(actions)
                pred, _ = model(images, actions, None)
                pred = pred.squeeze()
                pred = pred.permute(0, 2, 3, 1).data.numpy()

                dirname = 'videos/fwd_model/action2/pred{:d}'.format(vid)
                os.system("mkdir -p " + dirname)

                for t in range(opt.npred):
                    scipy.misc.imsave(dirname + '/im{:05d}.png'.format(t), pred[t])

                vid += 1

            pdb.set_trace()

            print(f'dv = {action.data[0][0][2]}, (dx, dy) = ({action.data[0][0][0]}, {action.data[0][0][1]}')
            action = action.data[0][0].numpy()
        else:
            action = None
        state, reward, done, vehicles = env.step(action)
        env.render()


for i in range(opt.n_episodes):
    print(f'episode {i + 1}/{opt.n_episodes}')
    runs = run_episode()
