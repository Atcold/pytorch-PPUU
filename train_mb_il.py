import torch, numpy, argparse, pdb, os, time, math
import utils
from dataloader import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import importlib

import models2 as models

#################################################
# Train an action-conditional forward model
#################################################

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-model', type=str, default='fwd-cnn')
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v2/')
parser.add_argument('-ncond', type=int, default=20)
parser.add_argument('-npred', type=int, default=20)
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-nfeature', type=int, default=128)
parser.add_argument('-n_hidden', type=int, default=128)
parser.add_argument('-beta', type=float, default=0.0, help='weight coefficient of prior loss')
parser.add_argument('-p_dropout', type=float, default=0.0, help='set z=0 with this probability')
parser.add_argument('-nz', type=int, default=2)
parser.add_argument('-n_mixture', type=int, default=10)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-grad_clip', type=float, default=1.0)
parser.add_argument('-epoch_size', type=int, default=1000)
parser.add_argument('-zeroact', type=int, default=0)
parser.add_argument('-warmstart', type=int, default=0)
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ten-bsize=16-ncond=20-npred=20-lrt=0.0001-nhidden=100-nfeature=128-combine=add-nz=32-beta=0.0-dropout=0.5-gclip=1.0-warmstart=1.model')
parser.add_argument('-combine', type=str, default='add')
parser.add_argument('-debug', type=int, default=0)
opt = parser.parse_args()

os.system('mkdir -p ' + opt.model_dir)

dataloader = DataLoader(None, opt, opt.dataset)



opt.n_inputs = 4
opt.n_actions = 2
opt.height = 117
opt.width = 24
opt.h_height = 14
opt.h_width = 3
opt.hidden_size = opt.nfeature*opt.h_height*opt.h_width


# load the model
model = torch.load(opt.model_dir + opt.mfile)
model.create_policy_net(opt)
model.intype('gpu')

opt.model_file = f'{opt.model_dir}/policy_networks/mbil-nfeature={opt.nfeature}-nhidden={opt.n_hidden}-mfile={opt.mfile}'

optimizer = optim.Adam(model.policy_net.parameters(), opt.lrt)

    
# training and testing functions. We will compute several losses:
# loss_i: images
# loss_s: states
# loss_c: costs
# loss_p: prior (optional)

def compute_loss(targets, predictions, r=True):
    target_images, target_states, target_costs = targets
    pred_images, pred_states, pred_costs = predictions
    loss_i = F.mse_loss(pred_images, target_images, reduce=r)
    loss_s = F.mse_loss(pred_states, target_states, reduce=r)
    loss_c = F.mse_loss(pred_costs, target_costs, reduce=r)
    return loss_i, loss_s, loss_c

def train(nbatches, npred):
    model.train()
    model.policy_net.train()
    total_loss_i, total_loss_s, total_loss_c = 0, 0, 0
    for i in range(nbatches):
        optimizer.zero_grad()
        inputs, actions, targets = dataloader.get_batch_fm('train', npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        pred, _ = model.train_policy_net(inputs, targets)
        loss_i, loss_s, loss_c = compute_loss(targets, pred)
        loss = loss_i + loss_s #+ loss_c
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.policy_net.parameters(), opt.grad_clip)
        optimizer.step()
        total_loss_i += loss_i.data[0]
        total_loss_s += loss_s.data[0]
        total_loss_c += loss_c.data[0]
        del inputs, actions, targets

    total_loss_i /= nbatches
    total_loss_s /= nbatches
    total_loss_c /= nbatches
    return total_loss_i, total_loss_s, total_loss_c, None


def test(nbatches):
    model.eval()
    total_loss_i, total_loss_s, total_loss_c = 0, 0, 0
    for i in range(nbatches):
        inputs, actions, targets = dataloader.get_batch_fm('valid')
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        pred, _ = model.train_policy_net(inputs, targets)
        loss_i, loss_s, loss_c = compute_loss(targets, pred)
        total_loss_i += loss_i.data[0]
        total_loss_s += loss_s.data[0]
        total_loss_c += loss_c.data[0]
        del inputs, actions, targets

    total_loss_i /= nbatches
    total_loss_s /= nbatches
    total_loss_c /= nbatches
    return total_loss_i, total_loss_s, total_loss_c, None
        

print('[training]')
for i in range(100):
    t0 = time.time()
    train_losses = train(opt.epoch_size, opt.npred)
    valid_losses = test(int(opt.epoch_size / 2))
    model.intype('cpu')
    torch.save(model, opt.model_file + '.model')
    model.intype('gpu')
    log_string = f'step {(i+1)*opt.epoch_size} | '
    log_string += utils.format_losses(*train_losses, 'train')
    log_string += utils.format_losses(*valid_losses, 'valid')
    print(log_string)
    utils.log(opt.model_file + '.log', log_string)

