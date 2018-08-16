import torch, numpy, argparse, pdb, os, time, random
import utils
from dataloader import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from sklearn import decomposition
import models2 as models

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models/')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-ncond', type=int, default=20)
parser.add_argument('-nz', type=int, default=32)
parser.add_argument('-npred', type=int, default=50)
parser.add_argument('-nfeature', type=int, default=128)
parser.add_argument('-n_hidden', type=int, default=256)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-epoch_size', type=int, default=100)
parser.add_argument('-combine', type=str, default='add')
parser.add_argument('-n_mixture', type=int, default=10)
parser.add_argument('-debug', type=int, default=0)
parser.add_argument('-cuda', type=int, default=1)
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1-nz=32-beta=0.0-nmix=1-warmstart=1.model')
opt = parser.parse_args()

opt.n_actions = 2
opt.height = 117
opt.width = 24
opt.h_height = 14
opt.h_width = 3
opt.hidden_size = opt.nfeature*opt.h_height*opt.h_width

dataloader = DataLoader(None, opt, opt.dataset)


prior = models.PriorMDN(opt).cuda()
model = torch.load(opt.model_dir + opt.mfile)
model.intype('gpu')
optimizer = optim.Adam(prior.parameters(), 0.001)

mfile_prior = f'{opt.model_dir}/{opt.mfile}-nfeature={opt.nfeature}-nhidden={opt.n_hidden}-lrt={opt.lrt}-nmixture={opt.n_mixture}.prior'
print(f'[will save prior model as: {mfile_prior}]')


def train(nbatches):
    model.train()
    total_loss = 0
    for i in range(nbatches):
        optimizer.zero_grad()
        inputs, actions, targets = dataloader.get_batch_fm('train', opt.npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        loss = prior.forward_thru_model(model, inputs, actions, targets)
        loss.backward()
        gnorm=utils.grad_norm(prior)
        torch.nn.utils.clip_grad_norm(prior.parameters(), 10)
        if not numpy.isnan(gnorm):
            optimizer.step()
        total_loss += loss.data[0]
    return total_loss / nbatches

def test(nbatches):
    model.eval()
    total_loss = 0
    for i in range(nbatches):
        inputs, actions, targets = dataloader.get_batch_fm('valid', opt.npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        loss = prior.forward_thru_model(model, inputs, actions, targets)
        total_loss += loss.data[0]
    return total_loss / nbatches
    
print('[training]')
for i in range(500):
    loss_train = train(opt.epoch_size)
    loss_test = test(opt.epoch_size)
    log_string = f'epoch {i} | train loss: {loss_train:.5f}, test loss: {loss_test:.5f}'
    print(log_string)
    utils.log(mfile_prior + '.log', log_string)
    prior.cpu()
    torch.save(prior, mfile_prior)
    prior.cuda()




