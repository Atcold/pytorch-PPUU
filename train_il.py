import torch, numpy, argparse, pdb
import models, utils
from dataloader import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


###########################################
# Train an imitation learner model
###########################################

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-model', type=str, default='policy-cnn-vae')
parser.add_argument('-nshards', type=int, default=40)
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models')
parser.add_argument('-n_episodes', type=int, default=20)
parser.add_argument('-lanes', type=int, default=8)
parser.add_argument('-ncond', type=int, default=4)
parser.add_argument('-npred', type=int, default=10)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-nfeature', type=int, default=64)
parser.add_argument('-n_hidden', type=int, default=100)
parser.add_argument('-nz', type=int, default=2)
parser.add_argument('-beta', type=float, default=0.1)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-epoch_size', type=int, default=1000)
opt = parser.parse_args()

opt.model_dir += f'_{opt.nshards}-shards/'

data_file = f'{opt.data_dir}/traffic_data_lanes={opt.lanes}-episodes=*-seed=*.pkl'
dataloader = DataLoader(data_file, opt)

opt.model_file = f'{opt.model_dir}/model={opt.model}-bsize={opt.batch_size}-ncond={opt.ncond}-npred={opt.npred}-lrt={opt.lrt}-nhidden={opt.n_hidden}-nfeature={opt.nfeature}'

if opt.model == 'policy-cnn-vae':
    opt.model_file += f'-nz={opt.nz}-beta={opt.beta}'

print(f'will save model as {opt.model_file}')

opt.n_inputs = 4
opt.n_actions = 3


prev_model = f'/misc/vlgscratch4/LecunGroup/nvidia-collab/models_20-shards/model=policy-cnn-bsize=32-ncond={opt.ncond}-npred={opt.npred}-lrt=0.0001-nhidden=100-nfeature={opt.nfeature}.model'


if opt.model == 'policy-mlp':
    policy = models.PolicyMLP(opt)
elif opt.model == 'policy-vae':
    policy = models.PolicyVAE(opt)
elif opt.model == 'policy-cnn':
    policy = models.PolicyCNN(opt)
elif opt.model == 'policy-cnn-vae':
    policy = models.PolicyCNN_VAE(opt, prev_model)

policy.intype('gpu')

optimizer = optim.Adam(policy.parameters(), opt.lrt)

def train(nbatches):
    policy.train()
    total_loss_mse, total_loss_kl = 0, 0
    for i in range(nbatches):
        optimizer.zero_grad()
        images, states, actions = dataloader.get_batch_il('train')
        images = Variable(images)
        states = Variable(states)
        actions = Variable(actions)
        pred_a, loss_kl = policy(images, states, actions)
        loss_mse = F.mse_loss(pred_a, actions)
        loss = loss_mse + opt.beta*loss_kl.cuda()
        loss.backward()
        optimizer.step()
        total_loss_mse += loss_mse.data[0]
        total_loss_kl += loss_kl.data[0]
    return total_loss_mse / nbatches, total_loss_kl / nbatches

def test(nbatches):
    policy.eval()
    total_loss_mse, total_loss_kl = 0, 0
    for i in range(nbatches):
        images, states, actions = dataloader.get_batch_il('valid')
        images = Variable(images)
        states = Variable(states)
        actions = Variable(actions)
        pred_a, loss_kl = policy(images, states, actions)
        loss_mse = F.mse_loss(pred_a, actions)
        total_loss_mse += loss_mse.data[0]
        total_loss_kl += loss_kl.data[0]
    return total_loss_mse / nbatches, total_loss_kl / nbatches

print('[training]')
best_valid_loss_mse = 1e6
for i in range(100):
    train_loss_mse, train_loss_kl = train(opt.epoch_size)
    valid_loss_mse, valid_loss_kl = test(opt.epoch_size)
    if valid_loss_mse < best_valid_loss_mse:
        best_valid_loss_mse = valid_loss_mse
        policy.intype('cpu')
        torch.save(policy, opt.model_file + '.model')
        policy.intype('gpu')

    log_string = f'iter {opt.epoch_size*i} | train loss: [MSE: {train_loss_mse:.5f}, KL: {train_loss_kl:.5f}], test: [{valid_loss_mse:.5f}, KL: {valid_loss_kl:.5f}], best MSE loss: {best_valid_loss_mse:.5f}'
    print(log_string)
    utils.log(opt.model_file + '.log', log_string)
