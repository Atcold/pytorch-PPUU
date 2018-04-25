import torch, numpy, argparse, pdb, os
import utils
import models2 as models
from dataloader import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


###########################################
# Train an imitation learner model
###########################################

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-model', type=str, default='policy-cnn-mdn')
parser.add_argument('-nshards', type=int, default=40)
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/')
parser.add_argument('-n_episodes', type=int, default=20)
parser.add_argument('-lanes', type=int, default=8)
parser.add_argument('-ncond', type=int, default=10)
parser.add_argument('-npred', type=int, default=20)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-nfeature', type=int, default=128)
parser.add_argument('-n_hidden', type=int, default=100)
parser.add_argument('-n_mixture', type=int, default=10)
parser.add_argument('-nz', type=int, default=2)
parser.add_argument('-beta', type=float, default=0.1)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-warmstart', type=int, default=0)
parser.add_argument('-epoch_size', type=int, default=1000)
parser.add_argument('-combine', type=str, default='add')
parser.add_argument('-grad_clip', type=float, default=10)
parser.add_argument('-debug', type=int, default=0)
opt = parser.parse_args()

opt.model_dir += f'/dataset_{opt.dataset}_costs2/models_il/'

opt.n_actions = 2
opt.n_inputs = opt.ncond
if opt.dataset == 'simulator':
    opt.height = 97
    opt.width = 20
    opt.h_height = 12
    opt.h_width = 2
    opt.model_dir += f'_{opt.nshards}-shards/'

elif opt.dataset == 'i80':
    opt.height = 117
    opt.width = 24
    opt.h_height = 14
    opt.h_width = 3
    opt.hidden_size = opt.nfeature*opt.h_height*opt.h_width



if opt.dataset == 'simulator':
    opt.model_dir += f'_{opt.nshards}-shards/'
    data_file = f'{opt.data_dir}/traffic_data_lanes={opt.lanes}-episodes=*-seed=*.pkl'
else:
    data_file = None
os.system('mkdir -p ' + opt.model_dir)

dataloader = DataLoader(data_file, opt, opt.dataset)

opt.model_file = f'{opt.model_dir}/model={opt.model}-bsize={opt.batch_size}-ncond={opt.ncond}-npred={opt.npred}-lrt={opt.lrt}-nhidden={opt.n_hidden}-nfeature={opt.nfeature}-nmixture={opt.n_mixture}-gclip={opt.grad_clip}'

if 'vae' in opt.model or '-ae-' in opt.model:
    opt.model_file += f'-nz={opt.nz}-beta={opt.beta}'

print(f'will save model as {opt.model_file}')

if opt.warmstart == 0:
    prev_model = ''

policy = models.PolicyMDN(opt)
    

policy.intype('gpu')

optimizer = optim.Adam(policy.parameters(), opt.lrt)

def train(nbatches):
    policy.train()
    total_loss = 0
    for i in range(nbatches):
        optimizer.zero_grad()
        inputs, actions, targets = dataloader.get_batch_fm('train')
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        pi, mu, sigma = policy(inputs[0], inputs[1])
        loss = utils.mdn_loss_fn(pi, sigma, mu, actions.view(opt.batch_size, -1))
        loss.backward()
        torch.nn.utils.clip_grad_norm(policy.parameters(), opt.grad_clip)
        optimizer.step()
        total_loss += loss.data[0]
    return total_loss / nbatches

def test(nbatches):
    policy.eval()
    total_loss = 0
    for i in range(nbatches):
        inputs, actions, targets = dataloader.get_batch_fm('valid')
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        pi, mu, sigma = policy(inputs[0], inputs[1])
        loss = utils.mdn_loss_fn(pi, sigma, mu, actions.view(opt.batch_size, -1))
        total_loss += loss.data[0]
    return total_loss / nbatches




print('[training]')
best_valid_loss = 1e6
for i in range(100):
    train_loss = train(opt.epoch_size)
    valid_loss = test(opt.epoch_size)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        policy.intype('cpu')
        torch.save(policy, opt.model_file + '.model')
        policy.intype('gpu')

    log_string = f'iter {opt.epoch_size*i} | train loss: {train_loss:.5f}, valid: {valid_loss:.5f}, best valid loss: {best_valid_loss:.5f}'
    print(log_string)
    utils.log(opt.model_file + '.log', log_string)
