import argparse, pdb
import gym
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import torch
from torch.autograd import Variable
from gym.envs.registration import register

# Run imitation learners in the environment

parser = argparse.ArgumentParser()
parser.add_argument('-display', type=int, default=1)
parser.add_argument('-seed', type=int, default=9999)
parser.add_argument('-lanes', type=int, default=5)
parser.add_argument('-traffic_rate', type=int, default=1)
parser.add_argument('-n_episodes', type=int, default=10)
parser.add_argument('-ncond', type=int, default=10)
parser.add_argument('-npred', type=int, default=10)
parser.add_argument('-n_samples', type=int, default=1)
parser.add_argument('-log_dir', type=str, default='logs/')
parser.add_argument('-models_dir', type=str, default='./models_il/')
parser.add_argument('-v', type=str, default='0')
opt = parser.parse_args()

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

kwargs = {
    'display': opt.display,
    'nb_lanes': opt.lanes,
    'delta_t': 0.1,
    'fps': 60,
    'store': False,
    'state_image': False,
    # 'policy_type': 'imitation',
    'traffic_rate': opt.traffic_rate,
}

register(
    id='Traffic-v0',
    entry_point='traffic_gym:StatefulEnv',
    kwargs=kwargs,
)

register(
    id='Traffic-v2',
    entry_point='traffic_gym_v2:MergingMap',
    kwargs=kwargs,
)

env = gym.make('Traffic-v' + opt.v)

mfile = 'model=policy-cnn-mdn-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-nmixture=10-gclip=10.model'
policy = torch.load(f'{opt.models_dir}/' + mfile)
policy.intype('cpu')


def run_episode():
    action = np.array([0, 0, 1, 0, 0, 0])
    states_, actions_, rewards_ = [], [], []
    done = False

    state, objects = env.reset()
    env.set_policy(policy)
    action = None
    cntr = 0
    time_until_crashed = -1
    t = 0
    while True:
        if t > 20000000:
            v = None
            for v_ in vehicles:
                if v_._id == env.policy_car_id:
                    v = v_

            if v.crashed:
                time_until_crashed = t
                break

            images = torch.stack(v._states_image).permute(0, 3, 2, 1).float()
            images /= 255.0
            images = Variable(images[-opt.ncond:].clone().float().unsqueeze(0))
            states, masks, actions = prepare_trajectory(v._states, v._actions)
            states = Variable(states[-opt.ncond:, 0].clone().unsqueeze(0))
            masks = Variable(masks[-opt.ncond:].unsqueeze(0))
            if images.size(1) < opt.ncond:
                action_ = None
            else:
                if action is None or cntr == opt.npred:
                    '''
                    sampled_actions = []
                    for s in range(opt.n_samples):
                    action, _ = policy(images, states, None)
                    sampled_actions.append(action)
                    sampled_actions = torch.stack(sampled_actions).squeeze().data

                    fig = plt.figure()
                    ax = Axes3D(fig)
                    for s in range(opt.n_samples):
                    ax.scatter(range(0, opt.npred), sampled_actions[s, :, 1], sampled_actions[s, :, 2])
                    pdb.set_trace()
                    ax.set_xlabel('time')
                    ax.set_ylabel('angle')
                    ax.set_zlabel('speed')
                    plt.show()
                    '''

#                    print('sampling new action sequence')
#                    action, _ = policy(images, states, None, unnormalize=True)
#                    action *= Variable(a_std.view(1, 1, 3))
#                    action += Variable(a_mean.view(1, 1, 3))
                    cntr = 0
                action_ = action.data[0][cntr].numpy()
#                print(f'dv = {action_[2]:0.4f}, (dx, dy) = ({action_[0]:0.4f}, {action_[1]:0.4f})')
                cntr += 1
        else:
            action_ = None

        state, reward, vehicles = env.step(action_)
        env.render()
        t += 1


        '''
        try:
            state, reward, done, vehicles = env.step(action_)
            env.render()
        except:
            break
            '''

    return time_until_crashed


crash_times = []
os.system(f'mkdir -p {opt.log_dir}')
for i in range(opt.n_episodes):
    print(f'episode {i + 1}/{opt.n_episodes}')
    time_until_crashed = run_episode()
    print('time until crashed: ' + str(time_until_crashed))
    if time_until_crashed != -1:
        crash_times.append(time_until_crashed)
    torch.save(torch.Tensor(time_until_crashed), f'{opt.log_dir}/test_il_mfile={mfile}_seed={opt.seed}.pth')
