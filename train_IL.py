import torch, numpy, argparse, pdb, os, math
import utils
import models
from dataloader import DataLoader
import torch.nn.functional as F
import torch.optim as optim


###########################################
# Train an imitation learner model
###########################################

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-v', type=int, default=4)
parser.add_argument('-model', type=str, default='policy-il-mdn')
parser.add_argument('-layers', type=int, default=3)
parser.add_argument('-fmap_geom', type=int, default=1)
parser.add_argument('-model_dir', type=str, default='models/policy_networks/')
parser.add_argument('-n_episodes', type=int, default=20)
parser.add_argument('-ncond', type=int, default=10)
parser.add_argument('-npred', type=int, default=20)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-dropout', type=float, default=0.0)
parser.add_argument('-nfeature', type=int, default=256)
parser.add_argument('-n_hidden', type=int, default=256)
parser.add_argument('-n_mixture', type=int, default=10)
parser.add_argument('-nz', type=int, default=2)
parser.add_argument('-beta', type=float, default=0.1)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-warmstart', type=int, default=0)
parser.add_argument('-epoch_size', type=int, default=1000)
parser.add_argument('-combine', type=str, default='add')
parser.add_argument('-grad_clip', type=float, default=50)
parser.add_argument('-debug', action='store_true')
parser.add_argument('-enable_tensorboard', action='store_true',
                    help='Enables tensorboard logging.')
parser.add_argument('-tensorboard_dir', type=str, default='models/policy_networks',
                    help='path to the directory where to save tensorboard log. If passed empty path' \
                         ' no logs are saved.')
opt = parser.parse_args()


opt.n_actions = 2
opt.n_inputs = opt.ncond
opt.height = 117
opt.width = 24
opt.h_height = 14
opt.h_width = 3
opt.hidden_size = opt.nfeature*opt.h_height*opt.h_width



os.system('mkdir -p ' + opt.model_dir)

dataloader = DataLoader(None, opt, opt.dataset)

opt.model_file = f'{opt.model_dir}/model={opt.model}-bsize={opt.batch_size}-ncond={opt.ncond}-npred={opt.npred}-lrt={opt.lrt}-nhidden={opt.n_hidden}-nfeature={opt.nfeature}-nmixture={opt.n_mixture}-gclip={opt.grad_clip}-seed={opt.seed}'

if 'vae' in opt.model or '-ten-' in opt.model:
    opt.model_file += f'-nz={opt.nz}-beta={opt.beta}'

print(f'[will save model as: {opt.model_file}]')

if opt.warmstart == 0:
    prev_model = ''

policy = models.PolicyMDN(opt, npred=opt.npred)
policy.intype('gpu')

optimizer = optim.Adam(policy.parameters(), opt.lrt, eps=1e-3)

def train(nbatches):
    policy.train()
    total_loss, nb = 0, 0
    for i in range(nbatches):
        optimizer.zero_grad()
        inputs, actions, targets, _, _ = dataloader.get_batch_fm('train')
        pi, mu, sigma, _ = policy(inputs[0], inputs[1])
        loss = utils.mdn_loss_fn(pi, sigma, mu, actions.view(opt.batch_size, -1))
        if not math.isnan(loss.item()):
            loss.backward()
            if opt.grad_clip != -1:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), opt.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            nb += 1
        else:
            print('warning, NaN')
    return total_loss / nb

def test(nbatches):
    policy.eval()
    total_loss, nb = 0, 0
    for i in range(nbatches):
        inputs, actions, targets, _, _ = dataloader.get_batch_fm('valid')
        pi, mu, sigma, _ = policy(inputs[0], inputs[1])
        loss = utils.mdn_loss_fn(pi, sigma, mu, actions.view(opt.batch_size, -1))
        if not math.isnan(loss.item()):
            total_loss += loss.item()
            nb += 1
        else:
            print('warning, NaN')
    return total_loss / nb



writer = utils.create_tensorboard_writer(opt)

print('[training]')
best_valid_loss = 1e6
for i in range(200):
    train_loss = train(opt.epoch_size)
    valid_loss = test(opt.epoch_size)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        policy.intype('cpu')
        torch.save(policy, opt.model_file + '.model')
        policy.intype('gpu')

    if writer is not None:
        writer.add_scalar('Loss/train', train_loss, i)
        writer.add_scalar('Loss/valid', valid_loss, i)

    log_string = f'iter {opt.epoch_size*i} | train loss: {train_loss:.5f}, valid: {valid_loss:.5f}, best valid loss: {best_valid_loss:.5f}'
    print(log_string)
    utils.log(opt.model_file + '.log', log_string)

    if writer is not None:
        writer.close()
