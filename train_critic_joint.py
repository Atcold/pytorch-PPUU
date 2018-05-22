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
parser.add_argument('-epoch_size', type=int, default=100)
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
parser.add_argument('-n_mixture', type=int, default=20)
parser.add_argument('-sampling', type=str, default='pdf')
parser.add_argument('-eval_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models/eval_critics2/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models/')
#parser.add_argument('-mfile', type=str, default='model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=0.0001-warmstart=1.model')
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1-nz=32-beta=0.0-nmix=1-warmstart=1.model')

parser.add_argument('-cuda', type=int, default=1)
opt = parser.parse_args()

opt.eval_dir += 'joint/'
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


def load_model(mfile):
    print(f'[loading {opt.model_dir + mfile}]')
    model = torch.load(opt.model_dir + mfile)
    model.eval()
    if '-ae' in mfile:
        pzfile = opt.model_dir + opt.mfile + '_100000.pz'
        if os.path.isfile(pzfile):
            p_z = torch.load(pzfile)
            graph = torch.load(pzfile + '.graph')
            model.p_z = p_z
            model.knn_indx = graph.get('knn_indx')
            model.knn_dist = graph.get('knn_dist')
            model.opt.topz_sample = int(model.p_z.size(0)*opt.graph_density)
            mfile_prior = f'{opt.model_dir}/{mfile}-nfeature=128-lrt=0.0001-nmixture=20.prior'
            print(f'[loading prior model: {mfile_prior}]')
            model.prior = torch.load(mfile_prior).cuda()
    model.opt.npred = opt.npred
    print('[done]')
    if opt.cuda == 1:
        model.intype('gpu')
    return model

model_files = ['model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1-nz=32-beta=0.0-nmix=1-warmstart=1.model', 
               'model=fwd-cnn-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-nz=32-beta=0.0-warmstart=0.model', 
               'model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=0.0001-warmstart=1.model']

model_list = [load_model(mfile) for mfile in model_files]
critic = models.LSTMCritic(opt)
if opt.cuda == 1:
    critic.cuda()
dataloader = DataLoader(None, opt, opt.dataset)

print(f'will save as {opt.critic_file}')


ones = Variable(torch.ones(len(model_files)*opt.batch_size).cuda())
zeros = Variable(torch.zeros(opt.batch_size).cuda())
labels = torch.cat((ones, zeros), 0)
optimizer = optim.Adam(critic.parameters(), opt.lrt)


def prepare_batch(pred_list, targets):
    p_images, p_states, p_costs = [], [], []
    for pred in pred_list:
        p_image, p_state, p_cost = pred
        p_images.append(p_image)
        p_states.append(p_state)
        p_costs.append(p_cost)
    p_images = torch.cat(p_images)
    p_states = torch.cat(p_states)
    p_costs = torch.cat(p_costs)
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
    scores = torch.zeros(len(model_list))
    for i in range(n_batches):
        optimizer.zero_grad()
        inputs, actions, targets = dataloader.get_batch_fm('train', opt.npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        pred_list = []
        for model in model_list:
            pred, _ = model(inputs, actions, targets, sampling=opt.sampling)
            pred_list.append(pred)
        batch = prepare_batch(pred_list, targets)
        logits = critic(batch)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm(critic.parameters(), 10)
        optimizer.step()
        critic.hidden[0].detach()
        critic.hidden[1].detach()
        total_loss += loss.data[0]
        scores += F.sigmoid(logits[labels==1].view(len(model_list), opt.batch_size)).mean(1).data.cpu()
    return total_loss / n_batches, scores / n_batches


def test(n_batches):
    total_loss = 0
    nskip = 0
    scores = torch.zeros(len(model_list))    
    for i in range(n_batches):
        optimizer.zero_grad()
        inputs, actions, targets = dataloader.get_batch_fm('valid', opt.npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        pred_list = []
        for model in model_list:
            pred, _ = model(inputs, actions, targets, sampling=opt.sampling)
            pred_list.append(pred)
        batch = prepare_batch(pred_list, targets)
        logits = critic(batch)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        if math.isnan(loss.data[0]):
            nskip += 1
        else:
            total_loss += loss.data[0]
        scores += F.sigmoid(logits[labels==1].view(len(model_list), opt.batch_size)).mean(1).data.cpu()
    return total_loss / (n_batches-nskip), scores / (n_batches-nskip)


best_valid_loss = 1e6
train_loss_all = []
valid_loss_all = []
valid_scores_all = []
print('[training]')
for i in range(100):
    train_loss, train_scores = train(opt.epoch_size)
    valid_loss, valid_scores = test(opt.epoch_size)
    train_loss_all.append(train_loss)
    valid_loss_all.append(valid_loss)
    valid_scores_all.append(valid_scores)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save({'critic':critic, 'model_list': model_list}, opt.critic_file + '.model')
    log_string = f'step: {i*opt.epoch_size} | [train loss: {train_loss:.5f}, scores [{train_scores[0]:.5f}, {train_scores[1]:.5f}, {train_scores[2]:.5f}]], [valid loss: {valid_loss:.5f}, scores [{valid_scores[0]:.5f}, {valid_scores[1]:.5f}, {valid_scores[2]:.5f}]], best valid loss: {best_valid_loss:.5f}'
    print(log_string)
    utils.log(opt.critic_file + '.log', log_string)
    torch.save({'train_loss': train_loss_all, 'valid_loss': valid_loss_all, 'valid_scores': valid_scores_all, 'model_list': model_list}, opt.critic_file + '.curves')



