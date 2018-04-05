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

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-debug', type=int, default=0)
parser.add_argument('-batch_size', type=int, default=4)
parser.add_argument('-v', type=int, default=1)
parser.add_argument('-display', type=int, default=0)
parser.add_argument('-seed', type=int, default=9999)
parser.add_argument('-lanes', type=int, default=8)
parser.add_argument('-traffic_rate', type=int, default=15)
parser.add_argument('-n_episodes', type=int, default=1)
parser.add_argument('-ncond', type=int, default=10)
parser.add_argument('-npred', type=int, default=200)
parser.add_argument('-n_samples', type=int, default=10)
parser.add_argument('-sampling', type=str, default='pdf')
parser.add_argument('-usphere', type=int, default=0)
parser.add_argument('-eval_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models/eval/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/')
#parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-warmstart=1.model')
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-een-fp-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-warmstart=1.model')
parser.add_argument('-cuda', type=int, default=1)
#parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-zsphere=0-gclip=-1-warmstart=1.model')
opt = parser.parse_args()

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
model = torch.load(opt.model_dir + opt.mfile)

model.eval()
if opt.cuda == 1:
    model.intype('gpu')

dataloader = DataLoader(data_file, opt, opt.dataset)

def compute_pz(nbatches):
    model.p_z = []
    for j in range(nbatches):
        print('[estimating z distribution: {:2.1%}]'.format(float(j)/nbatches), end="\r")
        inputs, actions, targets, _, _ = dataloader.get_batch_fm('train', opt.npred, cuda=(opt.cuda==1))
        inputs = Variable(inputs, volatile=True)
        actions = Variable(actions, volatile=True)
        targets = Variable(targets, volatile=True)
        pred, loss_kl = model(inputs, actions, targets, save_z = True)


model.opt.npred = opt.npred

if '-ae' in opt.mfile:
    if opt.sampling != 'fp':
        p_model_file = opt.model_dir + opt.mfile + f'-loss={opt.sampling}-usphere={opt.usphere}-nfeature=96.prior'
        print(f'[loading prior model: {p_model_file}]')
        model.q_network = torch.load(p_model_file)
        if opt.cuda == 1:
            model.q_network.cuda()
    if not hasattr(model.opt, 'z_sphere'):
            model.opt.z_sphere = 0
    if not hasattr(model.opt, 'z_mult'):
        model.opt.z_mult = 0
    compute_pz(200)
    print('[done]')

n_batches = 200
loss = torch.zeros(n_batches, opt.batch_size, opt.n_samples)
if opt.cuda == 1:
    loss = loss.cuda()

dirname = f'{opt.eval_dir}/{opt.mfile}-npred={opt.npred}-nsample={opt.n_samples}-{opt.sampling}-usphere={opt.usphere}.eval/'
for i in range(n_batches):
    torch.cuda.empty_cache()
    inputs_, actions_, targets_, _, _ = dataloader.get_batch_fm('test', opt.npred, cuda=(opt.cuda==1))
    if i < 10:
        for b in range(opt.batch_size):
            dirname_movie = '{}/videos/x{:d}/y/'.format(dirname, i*opt.batch_size + b)
            print('[saving ground truth video: {}]'.format(dirname_movie))
            utils.save_movie(dirname_movie, targets_[b])

    for s in range(opt.n_samples):
        print('[batch {}, sample {}'.format(i, s), end="\r")
        inputs = Variable(inputs_, volatile=True)
        actions = Variable(actions_, volatile=True)
        targets = Variable(targets_, volatile=True)
        pred_, _ = model(inputs, actions, None, sampling=opt.sampling)
        loss_ = F.mse_loss(targets, pred_, reduce=False)
        loss[i, :, s] += loss_.mean(2).mean(2).mean(2).mean(1).data
        if i < 10 and s < 20:
            for b in range(opt.batch_size):
                pred_b = pred_[b].clone()
#                pred_b = pred_b.squeeze().permute(0, 2, 3, 1).data.cpu().numpy()
                dirname_movie = '{}/videos/x{:d}/z{:d}/'.format(dirname, i*opt.batch_size + b, s)
                print('[saving video: {}]'.format(dirname_movie), end="\r")
                utils.save_movie(dirname_movie, pred_b.data, smooth=False)
                '''
                os.system("mkdir -p " + dirname_movie)
                for t in range(opt.npred):
                    if t > 0:
                        p = (pred_b[t] + pred_b[t-1])/2
                    else:
                        p = pred_b[t]
                    scipy.misc.imsave(dirname_movie + '/im{:05d}.png'.format(t), p)
                '''
        del inputs, actions, targets, pred_

torch.save(loss.view(-1, opt.n_samples).cpu(), f'{dirname}/loss.pth')
