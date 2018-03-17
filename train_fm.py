import torch, numpy, argparse, pdb, os, time
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
parser.add_argument('-nshards', type=int, default=20)
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models')
parser.add_argument('-n_episodes', type=int, default=20)
parser.add_argument('-lanes', type=int, default=8)
parser.add_argument('-ncond', type=int, default=10)
parser.add_argument('-npred', type=int, default=20)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-nfeature', type=int, default=64)
parser.add_argument('-n_hidden', type=int, default=100)
parser.add_argument('-tie_action', type=int, default=0)
parser.add_argument('-beta', type=float, default=1.0)
parser.add_argument('-nz', type=int, default=2)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-epoch_size', type=int, default=2000)
parser.add_argument('-zeroact', type=int, default=1)
parser.add_argument('-warmstart', type=int, default=1)
opt = parser.parse_args()


opt.model_dir += f'_{opt.nshards}-shards/'
os.system('mkdir -p ' + opt.model_dir)

data_file = f'{opt.data_dir}/traffic_data_lanes={opt.lanes}-episodes=*-seed=*.pkl'

dataloader = DataLoader(data_file, opt)


opt.model_file = f'{opt.model_dir}/model={opt.model}-bsize={opt.batch_size}-ncond={opt.ncond}-npred={opt.npred}-lrt={opt.lrt}-nhidden={opt.n_hidden}-nfeature={opt.nfeature}-tieact={opt.tie_action}'

if opt.zeroact == 1:
    opt.model_file += '-zeroact'

if 'vae' in opt.model:
    opt.model_file += f'-nz={opt.nz}'
    opt.model_file += f'-beta={opt.beta}'
    opt.model_file += f'-warmstart={opt.warmstart}'

if 'een' in opt.model:
    opt.model_file += f'-nz={opt.nz}'
    opt.model_file += f'-warmstart={opt.warmstart}'

print(f'will save model as {opt.model_file}')

opt.n_inputs = 4
opt.n_actions = 2
opt.height = 97
opt.width = 20


if opt.warmstart == 1:
    prev_model = f'/misc/vlgscratch4/LecunGroup/nvidia-collab/models_20-shards/'
    prev_model += f'model=fwd-cnn-bsize=32-ncond={opt.ncond}-npred={opt.npred}-lrt=0.0001-nhidden=100-nfeature={opt.nfeature}-tieact=0.model'
else:
    prev_model = ''

if opt.model == 'fwd-cnn-vae-fp':
    model = models.FwdCNN_VAE_FP(opt, mfile=prev_model)
elif opt.model == 'fwd-cnn-vae-lp':
    model = models.FwdCNN_VAE_LP(opt, mfile=prev_model)
elif opt.model == 'fwd-cnn-een-lp':
    model = models.FwdCNN_EEN_LP(opt, mfile=prev_model)
elif opt.model == 'fwd-cnn-een-fp':
    model = models.FwdCNN_EEN_FP(opt, mfile=prev_model)
elif opt.model == 'fwd-cnn-ae-fp':
    model = models.FwdCNN_AE_FP(opt, mfile=prev_model)
elif opt.model == 'fwd-cnn':
    model = models.FwdCNN(opt)
elif opt.model == 'fwd-cnn2':
    model = models.FwdCNN2(opt)

model.intype('gpu')

optimizer = optim.Adam(model.parameters(), opt.lrt)

def train(nbatches, npred):
    model.train()
    total_loss_mse, total_loss_kl = 0, 0
    for i in range(nbatches):
        optimizer.zero_grad()
        t0 = time.time()
        inputs, actions, targets, _, _ = dataloader.get_batch_fm('train', npred)
        t = time.time()-t0
#        print(f'get_batch: {t}')
        t0 = time.time()
        inputs = Variable(inputs)
        actions = Variable(actions)
        targets = Variable(targets)
        if opt.zeroact == 1:
            actions.data.zero_()
        pred, loss_kl = model(inputs, actions, targets)
        loss_mse = F.mse_loss(pred, targets)
        loss = loss_mse + opt.beta*loss_kl.cuda()
        loss.backward()
        optimizer.step()
        t = time.time()-t0
#        print(f'update: {t}')
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
        if opt.zeroact == 1:
            actions.data.zero_()
        pred, loss_kl = model(inputs, actions, targets)
        loss_mse = F.mse_loss(pred, targets)
        total_loss_mse += loss_mse.data[0]
        total_loss_kl += loss_kl.data[0]
    return total_loss_mse / nbatches, total_loss_kl / nbatches


def een_compute_pz(nbatches):
    model.p_z = []
    for j in range(nbatches):
        inputs, actions, targets, _, _ = dataloader.get_batch_fm('train', opt.npred)
        inputs = Variable(inputs)
        actions = Variable(actions)
        targets = Variable(targets)
        pred, loss_kl = model(inputs, actions, targets, save_z = True)
        

print('[training]')
best_valid_loss_mse = 1e6
for i in range(100):
    t0 = time.time()
    train_loss_mse, train_loss_kl = train(opt.epoch_size, opt.npred)
    valid_loss_mse, valid_loss_kl = test(int(opt.epoch_size / 2))
    t = time.time() - t0
    if valid_loss_mse < best_valid_loss_mse:
        best_valid_loss_mse = valid_loss_mse                
        if 'een' in opt.model:
            een_compute_pz(500)
        model.intype('cpu')
        torch.save(model, opt.model_file + '.model')
        model.intype('gpu')

    
    log_string = f'iter {opt.epoch_size*i} | train loss: [MSE: {train_loss_mse:.5f}, KL: {train_loss_kl:.5f}], test loss: [{valid_loss_mse:.5f}, KL: {valid_loss_kl:.5f}], best loss: {best_valid_loss_mse:.5f}, time={t:.4f}'
    print(log_string)
    utils.log(opt.model_file + '.log', log_string)



