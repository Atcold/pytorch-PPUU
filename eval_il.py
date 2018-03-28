import argparse, pdb
import gym
import numpy as np
import os
import pickle
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from gym.envs.registration import register
import scipy.misc
from dataloader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-v', type=int, default=1)
parser.add_argument('-display', type=int, default=0)
parser.add_argument('-seed', type=int, default=9999)
parser.add_argument('-lanes', type=int, default=8)
parser.add_argument('-traffic_rate', type=int, default=15)
parser.add_argument('-n_episodes', type=int, default=1)
parser.add_argument('-ncond', type=int, default=10)
parser.add_argument('-npred', type=int, default=10)
parser.add_argument('-tie_action', type=int, default=0)
parser.add_argument('-sigmout', type=int, default=1)
parser.add_argument('-n_samples', type=int, default=10)
parser.add_argument('-eval_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models/eval/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/')
parser.add_argument('-mfile', type=str, default='model=policy-ae-bsize=32-ncond=10-npred=10-lrt=0.0001-nhidden=100-nfeature=64-nz=8.model')
opt = parser.parse_args()

opt.n_actions = 2
opt.n_inputs = opt.ncond


random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)


opt.model_dir += f'/dataset_{opt.dataset}/models'
if opt.dataset == 'simulator':
    opt.model_dir += f'_{opt.nshards}-shards/'
    data_file = f'{opt.data_dir}/traffic_data_lanes={opt.lanes}-episodes=*-seed=*.pkl'
else:
    data_file = None
opt.model_dir += '/'

print(f'[loading {opt.model_dir + opt.mfile}]')
policy = torch.load(opt.model_dir + opt.mfile)
policy.intype('gpu')

dataloader = DataLoader(data_file, opt, opt.dataset)

images, states, actions = dataloader.get_batch_il('train')
images = Variable(images)
states = Variable(states)
actions = Variable(actions)
pred_a, loss_kl = policy(images, states, actions, save_z=True)
        

def compute_pz(nbatches):
    policy.p_z = []
    for j in range(nbatches):
        images, states, actions = dataloader.get_batch_il('train') 
        images = Variable(images)
        states = Variable(states)
        actions = Variable(actions)
        pred_a, loss_kl = policy(images, states, actions, save_z=True)

pred_a = []
for i in range(100):
    pred_a_, loss_kl = policy(images, states, None)
    pred_a.append(pred_a_)
pred_a=torch.stack(pred_a)
    
if '-ae' in opt.mfile:
    print('[estimating z distribution]')
    compute_pz(200)


