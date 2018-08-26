import torch, numpy, argparse, pdb, os, time, math
import utils
from dataloader import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import models
import importlib


#################################################
# Train an action-conditional forward model
#################################################

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-v', type=int, default=4)
parser.add_argument('-model', type=str, default='fwd-cnn')
parser.add_argument('-policy', type=str, default='policy-ten')
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v6/policy_networks2/')
parser.add_argument('-ncond', type=int, default=20)
parser.add_argument('-npred', type=int, default=100)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-layers', type=int, default=3)
parser.add_argument('-nfeature', type=int, default=256)
parser.add_argument('-n_hidden', type=int, default=256)
parser.add_argument('-beta', type=float, default=0.0, help='weight coefficient of prior loss')
parser.add_argument('-p_dropout', type=float, default=0.0, help='set z=0 with this probability')
parser.add_argument('-nz', type=int, default=2)
parser.add_argument('-n_mixture', type=int, default=10)
parser.add_argument('-context_dim', type=int, default=2)
parser.add_argument('-actions_subsample', type=int, default=4)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-grad_clip', type=float, default=1.0)
parser.add_argument('-epoch_size', type=int, default=500)
parser.add_argument('-curriculum_length', type=int, default=16)
parser.add_argument('-zeroact', type=int, default=0)
parser.add_argument('-warmstart', type=int, default=0)
parser.add_argument('-targetprop', type=int, default=0)
parser.add_argument('-loss_c', type=int, default=1)
parser.add_argument('-lambda_c', type=float, default=0.0)
parser.add_argument('-lambda_h', type=float, default=0.0)
parser.add_argument('-lambda_lane', type=float, default=0.1)
parser.add_argument('-lrt_traj', type=float, default=0.5)
parser.add_argument('-niter_traj', type=int, default=20)
parser.add_argument('-gamma', type=float, default=1.0)
parser.add_argument('-mfile', type=str, default='mbil-nfeature=256-npred=100-lambdac=0.0-lambdah=0.0-lanecost=0.1-tprop=0-gamma=0.997-curr=10-subs=100-cdim=2-lossc=1-dropout=0.0-seed=1')
parser.add_argument('-combine', type=str, default='add')
parser.add_argument('-debug', type=int, default=0)
parser.add_argument('-test_only', type=int, default=0)
opt = parser.parse_args()

opt.n_inputs = 4
opt.n_actions = 2
opt.height = 117
opt.width = 24
opt.h_height = 14
opt.h_width = 3
opt.hidden_size = opt.nfeature*opt.h_height*opt.h_width


os.system('mkdir -p ' + opt.model_dir + '/policy_networks2/')

if opt.targetprop == 1:
    opt.model_file += f'-lrt={opt.lrt_traj}-niter={opt.niter_traj}'

opt.model_file = opt.model_dir + opt.mfile

checkpoint = torch.load(opt.model_file + '.model')
model = checkpoint['model']
opt.fmap_geom = 1
model.create_prior_net(opt)
optimizer = optim.Adam(model.prior_net.parameters(), opt.lrt)
opt.model_file += '.prior'
print(f'[will save as: {opt.model_file}]')

assert(opt.actions_subsample != -1)

model.intype('gpu')

dataloader = DataLoader(None, opt, opt.dataset)

def train(nbatches, npred):
    model.train()
    model.prior_net.train()
    total_loss = 0
    for i in range(nbatches):
        optimizer.zero_grad()
        inputs, actions, targets = dataloader.get_batch_fm('train', npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        loss = model.train_prior_net(inputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.prior_net.parameters(), opt.grad_clip)
        optimizer.step()
        total_loss += loss.data[0]
        del inputs, actions, targets
    total_loss /= nbatches
    return total_loss

def test(nbatches, npred):
    model.eval()
    model.prior_net.eval()
    total_loss = 0
    for i in range(nbatches):
        optimizer.zero_grad()
        inputs, actions, targets = dataloader.get_batch_fm('valid', npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        loss = model.train_prior_net(inputs, targets)
        total_loss += loss.data[0]
        del inputs, actions, targets
    total_loss /= nbatches
    return total_loss



        

    


print('[training]')
utils.log(opt.model_file + '.log', f'[job name: {opt.model_file}]')
n_iter = 0
             
for i in range(500):
    train_loss = train(opt.epoch_size, opt.npred)
    valid_loss = test(int(opt.epoch_size / 2), opt.npred)
    n_iter += opt.epoch_size
    model.intype('cpu')
    '''
    torch.save({'model': model, 
                'optimizer': optimizer.state_dict(),
                'opt': opt, 
                'npred': npred, 
                'n_iter': n_iter}, 
               opt.model_file + '.model')
    ''' # TODO
    model.intype('gpu')
    log_string = f'step {n_iter} | train loss: {train_loss} | valid loss: {valid_loss} '
    print(log_string)
#    utils.log(opt.model_file + '.log', log_string)

