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

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-debug', type=int, default=0)
parser.add_argument('-batch_size', type=int, default=4)
parser.add_argument('-v', type=int, default=4)
parser.add_argument('-display', type=int, default=0)
parser.add_argument('-seed', type=int, default=9999)
parser.add_argument('-lanes', type=int, default=8)
parser.add_argument('-traffic_rate', type=int, default=15)
parser.add_argument('-n_episodes', type=int, default=1)
parser.add_argument('-ncond', type=int, default=20)
parser.add_argument('-npred', type=int, default=200)
parser.add_argument('-n_batches', type=int, default=200)
parser.add_argument('-n_samples', type=int, default=10)
parser.add_argument('-n_action_seq', type=int, default=5)
parser.add_argument('-sampling', type=str, default='fp')
parser.add_argument('-noise', type=float, default=0.0)
parser.add_argument('-n_mixture', type=int, default=20)
parser.add_argument('-n_models', type=int, default=10)
parser.add_argument('-graph_density', type=float, default=0.001)
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v9/')
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ten3-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-nhidden=128-fgeom=1-zeroact=0-zmult=0-dropout=0.1-nz=32-beta=0.0-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model')
parser.add_argument('-cuda', type=int, default=1)
parser.add_argument('-save_video', type=int, default=1)
opt = parser.parse_args()

if 'zeroact=1' in opt.mfile:
    opt.zeroact = 1
else:
    opt.zeroact = 0

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

opt.save_video = (opt.save_video == 1)
opt.eval_dir = opt.model_dir + f'/eval/'


print(f'[loading {opt.model_dir + opt.mfile}]')
model = torch.load(opt.model_dir + opt.mfile)
if type(model) is dict: model=model['model']
model.disable_unet=False
stats = torch.load('/misc/vlgscratch4/LecunGroup/nvidia-collab/traffic-data-atcold/data_i80_v0/data_stats.pth')
model.stats = stats


model.eval()
if opt.cuda == 1:
    model.intype('gpu')


dirname = f'{opt.eval_dir}/{opt.mfile}-nbatches={opt.n_batches}-npred={opt.npred}-nsample={opt.n_samples}'
if '-ten' in opt.mfile:
    dirname += f'-sampling={opt.sampling}'
    if opt.sampling == 'knn':
        dirname += f'-density={opt.graph_density}'
    elif opt.sampling == 'pdf':
        dirname += f'-nmixture={opt.n_mixture}'
        mfile_prior = f'{opt.model_dir}/{opt.mfile}-nfeature=128-lrt=0.0001-nmixture={opt.n_mixture}.prior'
        print(f'[loading prior model: {mfile_prior}]')
        model.prior = torch.load(mfile_prior).cuda()
dirname += '.eval'
os.system('mkdir -p ' + dirname)


dataloader = DataLoader(None, opt, opt.dataset)
model.opt.npred = opt.npred


if '-ten' in opt.mfile:
    pzfile = opt.model_dir + opt.mfile + '.pz'
    if os.path.isfile(pzfile):
        p_z = torch.load(pzfile)
        graph = torch.load(pzfile + '.graph')
        model.p_z = p_z
        model.knn_indx = graph.get('knn_indx')
        model.knn_dist = graph.get('knn_dist')
        model.opt.topz_sample = int(model.p_z.size(0)*opt.graph_density)            
    else:
        model.compute_pz(dataloader, 250)
        torch.save(model.p_z, pzfile)
        model.compute_z_graph()
        torch.save({'knn_dist': model.knn_dist, 'knn_indx': model.knn_indx}, pzfile + '.graph')
    print('[done]')

if opt.cuda == 1:
    model.intype('gpu')

loss_i = torch.zeros(opt.n_batches, opt.batch_size, opt.n_samples, opt.npred)
loss_s = torch.zeros(opt.n_batches, opt.batch_size, opt.n_samples, opt.npred)
loss_c = torch.zeros(opt.n_batches, opt.batch_size, opt.n_samples, opt.npred)
true_costs = torch.zeros(opt.n_batches, opt.batch_size, opt.npred, 2)
pred_costs = torch.zeros(opt.n_batches, opt.batch_size, opt.n_samples, opt.npred, 2)
true_states = torch.zeros(opt.n_batches, opt.batch_size, opt.npred, 4)
pred_states = torch.zeros(opt.n_batches, opt.batch_size, opt.n_samples, opt.npred, 4)


def compute_loss(targets, predictions, r=True):
    pred_images, pred_states, pred_costs, _ = predictions
    target_images, target_states, target_costs = targets
    loss_i = F.mse_loss(pred_images, target_images, reduce=r)
    loss_s = F.mse_loss(pred_states, target_states, reduce=r)
    loss_c = F.mse_loss(pred_costs, target_costs, reduce=r)
    return loss_i, loss_s, loss_c

dataloader.random.seed(12345)




model.train()
ui_truth, ui_perm, ui_turn = [], [], []
us_truth, us_perm, us_turn = [], [], []

for i in range(opt.n_batches):
    print(i)
    torch.cuda.empty_cache()    
    inputs, actions, targets, ids, car_sizes = dataloader.get_batch_fm('train', opt.npred)
    inputs = utils.make_variables(inputs)
    targets = utils.make_variables(targets)
    actions = Variable(actions)
    input_images, input_states = inputs[0], inputs[1]

    u_i, u_s, u_c, _, _, _, _ = planning.compute_uncertainty_batch(model, input_images, input_states, actions, targets, car_sizes)
    pred, loss_p = model(inputs, actions, targets, z_dropout=0)
    ui_truth.append(u_i)
    us_truth.append(u_s)


    actions_perm = actions[torch.randperm(4)]
    u_i, u_s, u_c, _, _, _, _ = planning.compute_uncertainty_batch(model, input_images, input_states, actions_perm, targets, car_sizes)
    pred, loss_p = model(inputs, actions_perm, targets, z_dropout=0)
    ui_perm.append(u_i)
    us_perm.append(u_s)

    actions_turn = actions.clone()
    actions_turn.data[:, :, 0].fill_(1)
    actions_turn.data[:, :, 1].fill_(2)
    u_i, u_s, u_c, _, _, _, _ = planning.compute_uncertainty_batch(model, input_images, input_states, actions_turn, targets, car_sizes)
    pred, loss_p = model(inputs, actions_turn, targets, z_dropout=0)
    ui_turn.append(u_i)
    us_turn.append(u_s)


ui_truth = torch.stack(ui_truth)
ui_turn = torch.stack(ui_turn)
ui_perm = torch.stack(ui_perm)
ui_truth = ui_truth.view(-1, opt.npred).cpu().numpy()
ui_turn = ui_turn.view(-1, opt.npred).cpu().numpy()
ui_perm = ui_perm.view(-1, opt.npred).cpu().numpy()
us_truth = torch.stack(us_truth)
us_turn = torch.stack(us_turn)
us_perm = torch.stack(us_perm)
us_truth = us_truth.view(-1, opt.npred).cpu().numpy()
us_turn = us_turn.view(-1, opt.npred).cpu().numpy()
us_perm = us_perm.view(-1, opt.npred).cpu().numpy()


mean,low,hi=utils.mean_confidence_interval(ui_truth)
utils.plot_mean_and_CI(mean, low, hi, color_mean='magenta', color_shading='magenta')
#mean,low,hi=utils.mean_confidence_interval(us_perm)
#utils.plot_mean_and_CI(mean, low, hi, color_mean='cyan', color_shading='cyan')
mean,low,hi=utils.mean_confidence_interval(ui_turn)
utils.plot_mean_and_CI(mean, low, hi, color_mean='blue', color_shading='blue')




















