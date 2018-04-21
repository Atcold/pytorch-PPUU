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
parser.add_argument('-n_batches', type=int, default=200)
parser.add_argument('-n_samples', type=int, default=10)
parser.add_argument('-sampling', type=str, default='fp')
parser.add_argument('-topz_sample', type=int, default=100)
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/')
#parser.add_argument('-mfile', type=str, default='model=fwd-cnn-bsize=16-ncond=10-npred=20-lrt=0.0002-nhidden=100-nfeature=96-decoder=0-combine=add-warmstart=0.model')
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-vae-lp-bsize=16-ncond=10-npred=20-lrt=0.0002-nhidden=100-nfeature=96-decoder=0-combine=add-nz=32-beta=0.0001-warmstart=1.model')
parser.add_argument('-cuda', type=int, default=1)
parser.add_argument('-save_video', type=int, default=1)
opt = parser.parse_args()

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

opt.save_video = (opt.save_video == 1)
opt.model_dir += f'/dataset_{opt.dataset}_costs2/models/'
opt.eval_dir = opt.model_dir + f'/eval2/'

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
        inputs, actions, targets = dataloader.get_batch_fm('train', opt.npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = utils.Variable(actions)
        pred, loss_kl = model(inputs, actions, targets, save_z = True)
        del inputs, actions, targets


model.opt.npred = opt.npred

if '-ae' in opt.mfile:
    '''
    if opt.sampling != 'fp':
        p_model_file = opt.model_dir + opt.mfile + f'-loss={opt.sampling}-usphere={opt.usphere}-nfeature=96.prior'
        print(f'[loading prior model: {p_model_file}]')
        model.q_network = torch.load(p_model_file)
        if opt.cuda == 1:
            model.q_network.cuda()
    '''
    compute_pz(10)
        
    print('[done]')

loss_i = torch.zeros(opt.n_batches, opt.batch_size, opt.n_samples)
loss_s = torch.zeros(opt.n_batches, opt.batch_size, opt.n_samples)
loss_c = torch.zeros(opt.n_batches, opt.batch_size, opt.n_samples)
true_costs = torch.zeros(opt.n_batches, opt.batch_size, opt.npred, 2)
pred_costs = torch.zeros(opt.n_batches, opt.batch_size, opt.n_samples, opt.npred, 2)
dirname = f'{opt.eval_dir}/{opt.mfile}-nbatches={opt.n_batches}-npred={opt.npred}-nsample={opt.n_samples}'

if 'fwd-cnn-ae-' in opt.mfile:
    dirname += f'-sampling={opt.sampling}'
    if opt.sampling == 'pdf':
        dirname += f'-topz={opt.topz_sample}'
        model.opt.topz_sample = opt.topz_sample

dirname += '.eval/'
os.system('mkdir -p ' + dirname)


def compute_loss(targets, predictions, r=True):
    pred_images, pred_states, pred_costs = predictions
    target_images, target_states, target_costs = targets
    loss_i = F.mse_loss(pred_images, target_images, reduce=r)
    loss_s = F.mse_loss(pred_states, target_states, reduce=r)
    loss_c = F.mse_loss(pred_costs, target_costs, reduce=r)
    return loss_i, loss_s, loss_c

dataloader.random.seed(12345)

for i in range(opt.n_batches):
    torch.cuda.empty_cache()    
    inputs_, actions_, targets_ = dataloader.get_batch_fm('test', opt.npred)

    if i < 10 and opt.save_video:
        for b in range(opt.batch_size):
            dirname_movie = '{}/videos/x{:d}/y/'.format(dirname, i*opt.batch_size + b)
            print('[saving ground truth video: {}]'.format(dirname_movie))
            utils.save_movie(dirname_movie, targets_[0][b])

    for s in range(opt.n_samples):
        print('[batch {}, sample {}'.format(i, s), end="\r")
        inputs = utils.make_variables(inputs_)
        targets = utils.make_variables(targets_)
        actions = utils.Variable(actions_)

        pred_, _= model(inputs, actions, None, sampling=opt.sampling)
        loss_i_s, loss_s_s, loss_c_s = compute_loss(targets, pred_, r=False)
        loss_i[i, :, s] += loss_i_s.mean(2).mean(2).mean(2).mean(1).data.cpu()
        loss_s[i, :, s] += loss_s_s.mean(2).mean(1).data.cpu()
        loss_c[i, :, s] += loss_c_s.mean(2).mean(1).data.cpu()
        pred_costs[i, :, s].copy_(pred_[2].data)
        true_costs[i].copy_(targets[2].data)
        if i < 10 and s < 20 and opt.save_video:
            for b in range(opt.batch_size):
                pred_b = pred_[0][b].clone()
                dirname_movie = '{}/videos/x{:d}/z{:d}/'.format(dirname, i*opt.batch_size + b, s)
                print('[saving video: {}]'.format(dirname_movie), end="\r")
                utils.save_movie(dirname_movie, pred_b.data, smooth=False)
        del inputs, actions, targets, pred_

torch.save({'loss_i': loss_i, 
            'loss_s': loss_s, 
            'loss_c': loss_c, 
            'true_costs': true_costs, 
            'pred_costs': pred_costs}, 
           f'{dirname}/loss.pth')

os.system('tar -cvf {}.tar.gz {}'.format(dirname, dirname))
