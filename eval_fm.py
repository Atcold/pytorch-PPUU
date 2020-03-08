import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as fun

import utils
from dataloader import DataLoader

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-debug', action='store_true')
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
parser.add_argument('-graph_density', type=float, default=0.001)
parser.add_argument('-model_dir', type=str, default='models/')
M1 = 'model=fwd-cnn-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-gclip=5.0-' + \
     'warmstart=0-seed=1.step200000.model'
M2 = 'model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-' + \
     'beta=1e-06-zdropout=0.0-gclip=5.0-warmstart=1-seed=1.step200000.model'
M3 = 'model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-' + \
     'beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model'
M4 = 'model=fwd-cnn-ten3-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-nhidden=128-fgeom=1-' + \
     'zeroact=0-zmult=0-dropout=0.1-nz=32-beta=0.0-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model'
parser.add_argument('-mfile', type=str, default=M3)
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
opt.eval_dir = opt.model_dir + f'eval/'


print(f'[loading {opt.model_dir + opt.mfile}]')
model = torch.load(opt.model_dir + opt.mfile)
if type(model) is dict: model = model['model']
model = model.cuda()
model.eval()
# if opt.cuda == 1:
    # model.intype('gpu')

dataloader = DataLoader(None, opt, opt.dataset)
# model.opt.npred = opt.npred  # instruct the model about how many predictions we want it to produce
model.opt.alpha = 0

dirname = f'{opt.eval_dir}{opt.mfile}-nbatches={opt.n_batches}-npred={opt.npred}-nsample={opt.n_samples}'
if '-ten' in opt.mfile:
    dirname += f'-sampling={opt.sampling}'
    if opt.sampling == 'knn':
        dirname += f'-density={opt.graph_density}'
    elif opt.sampling == 'pdf':
        dirname += f'-nmixture={opt.n_mixture}'
        mfile_prior = f'{opt.model_dir}/{opt.mfile}-nfeature=128-lrt=0.0001-nmixture={opt.n_mixture}.prior'
        print(f'[loading prior model: {mfile_prior}]')
        model.prior = torch.load(mfile_prior).cuda()
    # load z vectors. Extract them if they are not already saved.
    pzfile = opt.model_dir + opt.mfile + '.pz'
    if os.path.isfile(pzfile):
        p_z = torch.load(pzfile)
        graph = torch.load(pzfile + '.graph')
        model.p_z = p_z
        model.knn_indx = graph.get('knn_indx')
        model.knn_dist = graph.get('knn_dist')
        model.opt.topz_sample = int(model.p_z.size(0) * opt.graph_density)
    else:
        model.compute_pz(dataloader, opt, 250)
        torch.save(model.p_z, pzfile)
        model.compute_z_graph()
        torch.save({'knn_dist': model.knn_dist, 'knn_indx': model.knn_indx}, pzfile + '.graph')
    print('[done]')

dirname += '.eval'
os.system('mkdir -p ' + dirname)

# if opt.cuda == 1:
#     model.intype('gpu')

loss_i = torch.zeros(opt.n_batches, opt.batch_size, opt.n_samples, opt.npred)
loss_s = torch.zeros(opt.n_batches, opt.batch_size, opt.n_samples, opt.npred)
loss_c = torch.zeros(opt.n_batches, opt.batch_size, opt.n_samples, opt.npred)
true_costs = torch.zeros(opt.n_batches, opt.batch_size, opt.npred, 2)
pred_costs = torch.zeros(opt.n_batches, opt.batch_size, opt.n_samples, opt.npred, 2)
true_states = torch.zeros(opt.n_batches, opt.batch_size, opt.npred, 4)
pred_states = torch.zeros(opt.n_batches, opt.batch_size, opt.n_samples, opt.npred, 4)


def compute_loss(targets, predictions, r=True):
    pred_images, pred_states, _ = predictions
    target_images, target_states, target_costs = targets
    loss_i = fun.mse_loss(pred_images, target_images, reduce=r)
    loss_s = fun.mse_loss(pred_states, target_states, reduce=r)
    loss_c = fun.mse_loss(pred_costs.cuda(), target_costs.cuda(), reduce=r)
    return loss_i, loss_s, loss_c


dataloader.random.seed(12345)

for i in range(opt.n_batches):
    with torch.no_grad():
        torch.cuda.empty_cache()
        inputs, actions, targets, _, _ = dataloader.get_batch_fm('test', opt.npred)

        # save ground truth for the first 10 x batch_size samples
        if i < 10 and opt.save_video:
            for b in range(opt.batch_size):
                dirname_movie = f'{dirname}/videos/x{i * opt.batch_size + b:d}/y/'
                print(f'[saving ground truth video: {dirname_movie}]')
                utils.save_movie(dirname_movie, targets[0][b], targets[1][b], targets[2][b])

        for s in range(opt.n_samples):
            print(f'[batch {i}, sample {s}]', end="\r")

            if opt.zeroact == 1:
                actions.data.zero_()

            pred, _ = model(inputs, actions, targets, sampling=opt.sampling)  # return as many predictions as actions
            pred_states[i, :, s].copy_(pred[1])
            true_states[i].copy_(targets[1])

            if i < 10 and s < 20 and opt.save_video:
                for b in range(opt.batch_size):
                    dirname_movie = f'{dirname}/videos/sampled_z/true_actions/x{i * opt.batch_size + b:d}/z{s:d}/'
                    print(f'[saving video: {dirname_movie}]', end="\r")
                    utils.save_movie(dirname_movie, pred[0][b], pred[1][b])  # , pred_[2][b])
                    #                                    ^ images    ^ position and velocity

            # rotate actions across the batch: a_{t} -> a_{t + 1}
            actions_rot = actions[(torch.arange(opt.batch_size) - 1) % opt.batch_size]

            # also generate videos with different action sequences
            pred_rot, _ = model(inputs, actions_rot, targets, sampling=opt.sampling)
            if i < 10 and s < 20 and opt.save_video:
                for b in range(opt.batch_size):
                    dirname_movie = f'{dirname}/videos/sampled_z/rot_actions/x{i * opt.batch_size + b:d}/z{s:d}/'
                    print('[saving video: {}]'.format(dirname_movie), end="\r")
                    utils.save_movie(dirname_movie, pred_rot[0][b], pred_rot[1][b])  # , pred_perm[2][b])

            # also generate videos with true z vectors
            if s == 0:
                pred_true_z, _ = model(inputs, actions, targets)
                for b in range(opt.batch_size):
                    dirname_movie = f'{dirname}/videos/true_z/true_actions/x{i * opt.batch_size + b:d}/z{s:d}/'
                    print('[saving video: {}]'.format(dirname_movie), end="\r")
                    utils.save_movie(dirname_movie, pred_true_z[0][b], pred_true_z[1][b])  # , pred_true_z[2][b])

                pred_true_z_rot, _ = model(inputs, actions_rot, targets)
                for b in range(opt.batch_size):
                    dirname_movie = f'{dirname}/videos/true_z/rot_actions/x{i * opt.batch_size + b:d}/z{s:d}/'
                    print('[saving video: {}]'.format(dirname_movie), end="\r")
                    utils.save_movie(dirname_movie, pred_true_z_rot[0][b], pred_true_z_rot[1][b])
                    # , pred_true_z_perm[2][b])

            # del inputs, actions, targets, pred

torch.save({'loss_i': loss_i,
            'loss_s': loss_s,
            'loss_c': loss_c,
            'true_costs': true_costs,
            'pred_costs': pred_costs,
            'true_states': true_states,
            'pred_states': pred_states},
           f'{dirname}/loss.pth')

os.system(f'tar -cvf {dirname}.tgz {dirname}')
