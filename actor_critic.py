import argparse, os, random, numpy, pdb, utils
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import models

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-map', type=str, default='i80')
parser.add_argument('-v', type=int, default=4)
parser.add_argument('-model', type=str, default='fwd-cnn')
parser.add_argument('-policy', type=str, default='policy-gauss')
parser.add_argument('-ncond', type=int, default=20)
parser.add_argument('-npred', type=int, default=20)
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v11/')
parser.add_argument('-layers', type=int, default=3)
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-nfeature', type=int, default=256)
parser.add_argument('-n_hidden', type=int, default=256)
parser.add_argument('-dropout', type=float, default=0.0, help='regular dropout')
parser.add_argument('-lrt', type=float, default=0.00001)
parser.add_argument('-grad_clip', type=float, default=50.0)
parser.add_argument('-epoch_size', type=int, default=500)
parser.add_argument('-gamma', type=float, default=0.99)
parser.add_argument('-debug', action='store_true')
parser.add_argument('-n_env', type=int, default=8)
opt = parser.parse_args()

opt.n_inputs = 4
opt.n_actions = 2
opt.height = 117
opt.width = 24
opt.h_height = 14
opt.h_width = 3
opt.hidden_size = opt.nfeature*opt.h_height*opt.h_width


os.system('mkdir -p ' + opt.model_dir + '/policy_networks/')

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

gym.envs.registration.register(
    id='I-80-v1',
    entry_point='map_i80_ctrl:ControlledI80',
    kwargs={'fps': 10, 'nb_states': opt.ncond, 'display': 0, 'delta_t': 0.1},
)

print('Building the environment (loading data, if any)')
env_names = {
    'i80': 'I-80-v1',
}

env = gym.make(env_names[opt.map])

env_list = []
for k in range(opt.n_env):
    print(f'[initializing environment {k}]')
    env_list.append(gym.make(env_names[opt.map]))

stats = torch.load('/misc/vlgscratch4/LecunGroup/nvidia-collab/traffic-data-atcold/data_i80_v0/data_stats.pth')


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

model = models.StochasticPolicy(opt, context_dim=0, actor_critic=True).cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lrt)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    img, vec = state['context'], state['state']
    img = img.unsqueeze(0).contiguous().float().div_(255).cuda()
    vec = vec.unsqueeze(0).contiguous().float().cuda()
    vec = vec - stats['s_mean'].unsqueeze(0).cuda()
    vec = vec / stats['s_std'].unsqueeze(0).cuda()
    action, _, mu, std, state_value = model(img, vec)
    model.saved_actions.append(SavedAction(utils.log_pdf(action, mu, std), state_value[0]))
    return action.data


def select_action_batch(states):
    imgs, vecs = [], []
    for k in range(len(states)):
        imgs.append(states[k]['context'])
        vecs.append(states[k]['state'])
    imgs = torch.stack(imgs)
    vecs = torch.stack(vecs)
    imgs = imgs.contiguous().float().div_(255).cuda()
    vecs = vecs.contiguous().float().cuda()
    vec = vec - stats['s_mean'].unsqueeze(0).expand(vecs.size()).cuda()
    vec = vec / stats['s_std'].unsqueeze(0).expand(vecs.size()).cuda()
    actions, _, mu, std, state_values = model(imgs, vecs)
    model.saved_actions.append(SavedAction(utils.log_pdf(actions, mu, std), state_values))
    return action.data


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + opt.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards).cuda()
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r]).cuda()))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    print(utils.grad_norm(model).item())
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]







def main_parallel():
    running_reward = 10
    for i_episode in count(1):
        states = []
        for k in range(opt.n_env):
            states.append(env.reset)
        pdb.set_trace()
        states = torch.stack(states)
        for t in range(10000):  # Don't infinite loop while learning
            pdb.set_trace()
            actions = select_action(states)
            actions = actions * stats['a_std'].cuda()
            actions = actions + stats['a_mean'].cuda()
            state, reward, done, _ = env.step(action.cpu().numpy())
            if reward['collisions_per_frame'] > 0:
                print('[collision]')
                done = True
            print(action, [reward['pixel_proximity_cost'], reward['lane_cost']])
            reward = 1.2-torch.tensor((reward['pixel_proximity_cost'] + 0.2*reward['lane_cost'])).cuda()
            model.rewards.append(reward)
            if done:
                print(f'ending after {t} timesteps')
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % 10 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))






def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            action = action * stats['a_std'].cuda()
            action = action + stats['a_mean'].cuda()
            state, reward, done, _ = env.step(action.cpu().numpy())
            if reward['collisions_per_frame'] > 0:
                print('[collision]')
                done = True
            print(action, [reward['pixel_proximity_cost'], reward['lane_cost']])
            reward = 1.2-torch.tensor((reward['pixel_proximity_cost'] + 0.2*reward['lane_cost'])).cuda()
            model.rewards.append(reward)
            if done:
                print(f'ending after {t} timesteps')
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % 10 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))


if __name__ == '__main__':
    main()
