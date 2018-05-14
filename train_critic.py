import argparse, pdb, os, pickle, random, sys, math
import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from gym.envs.registration import register
import scipy.misc
from dataloader import DataLoader
import utils
import models2 as models

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
parser.add_argument('-nfeature', type=int, default=128)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-nhidden', type=int, default=128)
parser.add_argument('-combine', type=str, default='add')
parser.add_argument('-n_samples', type=int, default=10)
parser.add_argument('-graph_density', type=float, default=0.005)
parser.add_argument('-sampling', type=str, default='pdf')
parser.add_argument('-eval_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models/eval_critics/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models/')
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=0.0001-warmstart=1.model')
parser.add_argument('-cuda', type=int, default=1)
opt = parser.parse_args()

opt.eval_dir += opt.mfile
os.system('mkdir -p ' + opt.eval_dir)
opt.critic_file = opt.eval_dir + f'/critic-nfeature={opt.nfeature}-nhidden={opt.nhidden}-lrt={opt.lrt}-sampling={opt.sampling}-seed={opt.seed}.model'

if opt.dataset == 'simulator':
    opt.height = 97
    opt.width = 20
    opt.h_height = 12
    opt.h_width = 2

elif opt.dataset == 'i80':
    opt.height = 117
    opt.width = 24
    opt.h_height = 14
    opt.h_width = 3
    opt.hidden_size = opt.nfeature*opt.h_height*opt.h_width

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)



print(f'[loading {opt.model_dir + opt.mfile}]')
model = torch.load(opt.model_dir + opt.mfile)
model.eval()
critic = models.LSTMCritic(opt)
if opt.cuda == 1:
    model.intype('gpu')
    critic.cuda()

dataloader = DataLoader(None, opt, opt.dataset)

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
    pzfile = opt.model_dir + opt.mfile + '_100000.pz'
    if os.path.isfile(pzfile):
        p_z = torch.load(pzfile)
        graph = torch.load(pzfile + '.graph')
        model.p_z = p_z
        model.knn_indx = graph.get('knn_indx')
        model.knn_dist = graph.get('knn_dist')
        model.opt.topz_sample = int(model.p_z.size(0)*opt.graph_density)
    if opt.sampling != 'fp':
        opt.critic_file += f'-density={opt.graph_density}'
            
    print('[done]')

print(f'will save as {opt.critic_file}')
n_batches = 200
loss = torch.zeros(n_batches, opt.batch_size, opt.n_samples)
if opt.cuda == 1:
    loss = loss.cuda()


ones = Variable(torch.ones(opt.batch_size).cuda())
zeros = Variable(torch.zeros(opt.batch_size).cuda())
labels = torch.cat((ones, zeros), 0)
optimizer = optim.Adam(critic.parameters(), opt.lrt)


def prepare_batch(pred, targets):
    p_images, p_states, p_costs = pred
    t_images, t_states, t_costs = targets
    images = torch.cat((p_images, t_images), 0)
    states = torch.cat((p_states, t_states), 0)
    costs = torch.cat((p_costs, t_costs), 0)
    states_costs = torch.cat((states, costs), 2)
    images.detach()
    states_costs.detach()
    return [images, states_costs]

def train(n_batches):
    total_loss = 0
    for i in range(n_batches):
        optimizer.zero_grad()
        inputs, actions, targets = dataloader.get_batch_fm('train', opt.npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        pred, loss_p = model(inputs, actions, targets, sampling=opt.sampling)
        batch = prepare_batch(pred, targets)
        logits = critic(batch)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        critic.hidden[0].detach()
        critic.hidden[1].detach()
        total_loss += loss.data[0]
    return total_loss / n_batches


def test(n_batches):
    total_loss = 0
    nskip = 0
    for i in range(n_batches):
        optimizer.zero_grad()
        inputs, actions, targets = dataloader.get_batch_fm('valid', opt.npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        pred, _ = model(inputs, actions, targets, sampling=opt.sampling)
        batch = prepare_batch(pred, targets)
        logits = critic(batch)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        if math.isnan(loss.data[0]):
            nskip += 1
        else:
            total_loss += loss.data[0]
    return total_loss / (n_batches-nskip)


best_valid_loss = 1e6
train_loss_all = []
valid_loss_all = []
print('[training]')
for i in range(100):
    train_loss = train(20)
    valid_loss = test(100)
    train_loss_all.append(train_loss)
    valid_loss_all.append(valid_loss)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(critic, opt.critic_file + '.model')
    log_string = f'epoch: {i+1} | train loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}, best valid loss: {best_valid_loss:.5f}'
    print(log_string)
    utils.log(opt.critic_file + '.log', log_string)
    torch.save({'train_loss': train_loss_all, 'valid_loss': valid_loss_all}, opt.critic_file + '.curves')



