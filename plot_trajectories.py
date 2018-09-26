import torch, numpy, argparse, pdb, os, time, math, random, copy
import utils
from dataloader import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import models, planning
import importlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-model', type=str, default='fwd-cnn')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v11/planning_results2/')
opt = parser.parse_args()

'''
path1='policy-il-model=policy-il-mdn-bsize=64-ncond=20-npred=1-lrt=0.0001-nhidden=256-nfeature=256-nmixture=1-gclip=50.model.actions'
path2='policy-svg-svg-policy-gauss-model=vae-zdropout=0.0-policy-gauss-nfeature=256-npred=20-ureg=0.01-lambdal=0.2-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=1-seed=1-novalue.model.actions'
path3='policy-svg-svg-policy-gauss-model=vae-zdropout=0.0-policy-gauss-nfeature=256-npred=20-ureg=0.05-lambdal=0.2-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=1-seed=1-novalue.model.actions'
'''

path1='policy-tm-mbil-policy-gauss-nfeature=256-npred=1-lambdac=0.0-gamma=0.99-seed=1-deterministic.model.states'
#path2='policy-tm-mbil-policy-gauss-nfeature=256-npred=3-lambdac=0.0-gamma=0.99-seed=1-deterministic.model.states'
path2='policy-svg-svg-policy-gauss-model=vae-zdropout=0.0-policy-gauss-nfeature=256-npred=20-ureg=0.01-lambdal=0.2-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=1-seed=1-novalue.model.states'
path3='policy-tm-mbil-policy-gauss-nfeature=256-npred=20-lambdac=0.0-gamma=0.99-seed=1-deterministic.model.states'

a1 = torch.load(opt.model_dir + path1)
a2 = torch.load(opt.model_dir + path2)
a3 = torch.load(opt.model_dir + path3)
plt.ion()


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')


for k in range(20):
    '''
    a1_k = torch.tensor(a1[k]).squeeze()
    a2_k = torch.tensor(a2[k]).squeeze()
    a3_k = torch.tensor(a3[k]).squeeze()
    '''

    a1_k = torch.stack(a1[k]).squeeze()[:, -1, :2]
    a2_k = torch.stack(a2[k]).squeeze()[:, -1, :2]
    a3_k = torch.stack(a3[k]).squeeze()[:, -1, :2]

#    ax.plot(xs=a1_k[:, 0].numpy(), ys=a1_k[:, 1].numpy(), zs=range(a1_k.size(0)), c='red')
#    ax.plot(xs=a2_k[:, 0].numpy(), ys=a2_k[:, 1].numpy(), zs=range(a2_k.size(0)), c='blue')
#    ax.plot(xs=a3_k[:, 0].numpy(), ys=a3_k[:, 1].numpy(), zs=range(a3_k.size(0)), c='blue')

    k=4
    plt.plot(a1_k[::k, 0].numpy(), -a1_k[::k, 1].numpy(), '--', c='black', markersize=5)
    plt.plot(a2_k[::k, 0].numpy(), -a2_k[::k, 1].numpy(), '-', c='red', markersize=5)
    plt.plot(a3_k[::k, 0].numpy(), -a3_k[::k, 1].numpy(), '-', c='blue', markersize=5)
#    a3_k = torch.tensor(a3[k]).squeeze()
#    plt.plot(a3_k[:, 0].numpy(), a3_k[:, 1].numpy(), '-', c='green')

plt.legend(['1-step IL', 'MBIL (20 step)', 'SVG + Uncertainty Cost'], fontsize=10)
plt.xlabel('Direction of traffic', fontsize=16)
plt.xticks([])
plt.yticks([])
plt.savefig('plots/trajectories.pdf')
