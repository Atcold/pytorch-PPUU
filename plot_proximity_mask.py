import argparse, pdb, os, pickle, random, sys
import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from gym.envs.registration import register
import scipy.misc
from dataloader import DataLoader
import utils
import matplotlib.pyplot as plt
import planning
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-debug', type=int, default=0)
parser.add_argument('-seed', type=int, default=9999)
parser.add_argument('-ncond', type=int, default=20)
parser.add_argument('-npred', type=int, default=200)
opt = parser.parse_args()

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

dataloader = DataLoader(None, opt, opt.dataset)


inputs, actions, targets, ids, car_sizes = dataloader.get_batch_fm('train')
inputs = utils.make_variables(inputs)
images, states = inputs[0], inputs[1]
_, mask = utils.proximity_cost(images, states, car_size=car_sizes, unnormalize=True, s_mean=dataloader.s_mean, s_std=dataloader.s_std)

states2 = states[:, -1].clone()
states2 = states2 * (1e-8 + dataloader.s_std.view(1, 4)).cuda()
states2 = states2 + dataloader.s_mean.view(1, 4).cuda()

plt.imshow(images[1][-1].permute(1, 2, 0))
plt.axis('off')
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
plt.axis('off')
plt.savefig('plots/image_503.pdf')
plt.close()
plt.imshow(mask[1][0])
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
plt.axis('off')
plt.savefig('plots/mask_503.pdf')
plt.close()
plt.imshow(images[0][-1].permute(1, 2, 0))
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
plt.axis('off')
plt.savefig('plots/image_198.pdf')
plt.close()
plt.imshow(mask[0][0])
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
plt.axis('off')
plt.savefig('plots/mask_198.pdf')


