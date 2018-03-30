import argparse, pdb, os, pickle, random
import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
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
parser.add_argument('-npred', type=int, default=200)
parser.add_argument('-tie_action', type=int, default=0)
parser.add_argument('-sigmout', type=int, default=1)
parser.add_argument('-n_samples', type=int, default=10)
parser.add_argument('-sampling', type=str, default='sphere')
parser.add_argument('-usphere', type=int, default=1)
parser.add_argument('-eval_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models/eval/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/')
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-zsphere=1-gclip=-1-warmstart=1.model')
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
model.q_network = torch.load(opt.model_dir + opt.mfile + f'-loss={opt.sampling}-usphere={opt.usphere}-nfeature=64.prior')
model.intype('gpu')
model.q_network.cuda()

dataloader = DataLoader(data_file, opt, opt.dataset)

def compute_pz(nbatches):
    model.p_z = []
    for j in range(nbatches):
        inputs, actions, targets, _, _ = dataloader.get_batch_fm('train', opt.npred)
        inputs = Variable(inputs)
        actions = Variable(actions)
        targets = Variable(targets)
        pred, loss_kl = model(inputs, actions, targets, save_z = True)



'''
inputs, actions, targets, _, _ = dataloader.get_batch_fm('valid', opt.npred)
inputs = Variable(inputs)
actions = Variable(actions)
targets = Variable(targets)
pred, loss_kl = model(inputs, actions, None, sampling=opt.sampling)
'''

model.opt.npred = opt.npred

if '-ae' in opt.mfile:
    print('[estimating z distribution]')
    compute_pz(200)
    print('[done]')

n_batches = 100
loss = torch.zeros(n_batches, opt.batch_size, opt.n_samples).cuda()
dirname = f'{opt.eval_dir}/{opt.mfile}-npred={opt.npred}-nsample={opt.n_samples}-{opt.sampling}-usphere={opt.usphere}.eval/'
for i in range(n_batches):
    print(f'i={i}')
    inputs_, actions_, targets_, _, _ = dataloader.get_batch_fm('test', opt.npred)
    for s in range(opt.n_samples):
        print(f'sample {s}')
        inputs = Variable(inputs_.cuda())
        actions = Variable(actions_.cuda())
        target = Variable(targets_.cuda())
        pred, _ = model(inputs, actions, None, sampling=opt.sampling)
        loss_ = F.mse_loss(target, pred, reduce=False)
        loss[i, :, s] += loss_.mean(2).mean(2).mean(2).mean(1).data
        if i < 10:
            for b in range(opt.batch_size):
                pred_b = pred[b].clone()
                pred_b = pred_b.squeeze().permute(0, 2, 3, 1).data.cpu().numpy()
                dirname_movie = '{}/videos/x{:d}/z{:d}/'.format(dirname, i*opt.batch_size + b, s)
                os.system("mkdir -p " + dirname_movie)
                print(f'[saving video: {dirname_movie}]')
                for t in range(opt.npred):
                    if t > 0:
                        p = (pred_b[t] + pred_b[t-1])/2
                    else:
                        p = pred_b[t]
                    scipy.misc.imsave(dirname_movie + '/im{:05d}.png'.format(t), p)

torch.save(loss.view(-1, opt.n_samples).cpu(), f'{dirname}/loss.pth')
