import torch, numpy, argparse, pdb
import models, utils
from dataloader import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

#################################################
# Train an action-conditional forward model
#################################################

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-model', type=str, default='fwd-cnn')
parser.add_argument('-nshards', type=int, default=30)
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models/')
parser.add_argument('-n_episodes', type=int, default=100)
parser.add_argument('-lanes', type=int, default=3)
parser.add_argument('-ncond', type=int, default=4)
parser.add_argument('-npred', type=int, default=10)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-nfeature', type=int, default=64)
parser.add_argument('-n_hidden', type=int, default=100)
parser.add_argument('-nz', type=int, default=2)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-epoch_size', type=int, default=500)
opt = parser.parse_args()

data_file = f'{opt.data_dir}/traffic_data_lanes={opt.lanes}-episodes=*-seed=*.pkl'
dataloader = DataLoader(data_file, opt)

opt.model_file = f'{opt.model_dir}/model={opt.model}-ncond={opt.ncond}-npred={opt.npred}-lrt={opt.lrt}-nhidden={opt.n_hidden}-nfeature={opt.nfeature}'
print(f'will save model as {opt.model_file}')

opt.n_inputs = 4
opt.n_actions = 3

if opt.model == 'fwd-vae':
    model = models.FwdVAE(opt)
elif opt.model == 'fwd-cnn':
    model = models.FwdCNN(opt)

model.intype('gpu')

optimizer = optim.Adam(model.parameters(), opt.lrt)

def train(nbatches):
    model.train()
    total_loss_mse, total_loss_kl = 0, 0
    for i in range(nbatches):
        optimizer.zero_grad()
        inputs, actions, targets, _, _ = dataloader.get_batch_fm('train')
        inputs = Variable(inputs)
        actions = Variable(actions)
        targets = Variable(targets)
        pred, loss_kl = model(inputs, actions, targets)
        loss_mse = F.mse_loss(pred, targets)
        loss = loss_mse + loss_kl.cuda()
        loss.backward()
        optimizer.step()
        total_loss_mse += loss_mse.data[0]
        total_loss_kl += loss_kl.data[0]
    return total_loss_mse / nbatches, total_loss_kl / nbatches


def test(nbatches):
    model.eval()
    total_loss_mse, total_loss_kl = 0, 0
    for i in range(nbatches):
        inputs, actions, targets, _, _ = dataloader.get_batch_fm('valid')
        inputs = Variable(inputs)
        actions = Variable(actions)
        targets = Variable(targets)
        pred, loss_kl = model(inputs, actions, targets)
        loss_mse = F.mse_loss(pred, targets)
        total_loss_mse += loss_mse.data[0]
        total_loss_kl += loss_kl.data[0]
    return total_loss_mse / nbatches, total_loss_kl / nbatches

print('[training]')
for i in range(100):
    train_loss_mse, train_loss_kl = train(opt.epoch_size)
    valid_loss_mse, valid_loss_kl = test(opt.epoch_size)
    log_string = f'iter {opt.epoch_size*i} | train loss: [MSE: {train_loss_mse}, KL: {train_loss_kl}], test: [{valid_loss_mse}, KL: {valid_loss_kl}]'
    print(log_string)
    utils.log(opt.model_file + '.log', log_string)
    model.intype('cpu')
    torch.save(model, opt.model_file + '.model')
    model.intype('gpu')
