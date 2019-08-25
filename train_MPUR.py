import argparse
import math
import numpy
import os
import pdb
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from os import path

import planning
import utils
from dataloader import DataLoader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#################################################
# Train a policy / controller
#################################################

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-v', type=int, default=0)
parser.add_argument('-model', type=str, default='fwd-cnn')
parser.add_argument('-policy', type=str, default='policy-deterministic')
parser.add_argument('-model_dir', type=str, default='models/')
parser.add_argument('-ncond', type=int, default=20)
parser.add_argument('-npred', type=int, default=30)
parser.add_argument('-layers', type=int, default=3)
parser.add_argument('-batch_size', type=int, default=12)
parser.add_argument('-nfeature', type=int, default=256)
parser.add_argument('-n_hidden', type=int, default=256)
parser.add_argument('-dropout', type=float, default=0.0, help='regular dropout')
parser.add_argument('-lrt', type=float, default=0.0001, help='learning rate')
parser.add_argument('-grad_clip', type=float, default=50.0)
parser.add_argument('-epoch_size', type=int, default=500)
parser.add_argument('-n_futures', type=int, default=10)
parser.add_argument('-u_reg', type=float, default=0.05, help='coefficient of uncertainty regularization term')
parser.add_argument('-u_hinge', type=float, default=0.5)
parser.add_argument('-lambda_a', type=float, default=0.0, help='l2 regularization on actions')
parser.add_argument('-lambda_l', type=float, default=0.2, help='coefficient of lane cost')
parser.add_argument('-lrt_z', type=float, default=0.0)
parser.add_argument('-z_updates', type=int, default=0)
parser.add_argument('-infer_z', type=int, default=0)
parser.add_argument('-gamma', type=float, default=0.99)
parser.add_argument('-learned_cost', action='store_true')
M1 = 'model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-' + \
     'beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model'
M2 = 'model=fwd-cnn-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-gclip=5.0-' + \
     'warmstart=0-seed=1.step200000.model'
parser.add_argument('-mfile', type=str, default=M1, help='dynamics model used to train the policy network')
parser.add_argument('-value_model', type=str, default='')
parser.add_argument('-load_model_file', type=str, default='')
parser.add_argument('-combine', type=str, default='add')
parser.add_argument('-debug', action='store_true')
parser.add_argument('-save_movies', action='store_true')
parser.add_argument('-l2reg', type=float, default=0.0)
parser.add_argument('-no_cuda', action='store_true')
opt = parser.parse_args()

opt.n_inputs = 4
opt.n_actions = 2
opt.height = 117
opt.width = 24
opt.h_height = 14
opt.h_width = 3
opt.hidden_size = opt.nfeature * opt.h_height * opt.h_width

os.system('mkdir -p ' + path.join(opt.model_dir, 'policy_networks'))

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)  # I think it's no longer required

# Define default device
opt.device = torch.device("cuda" if torch.cuda.is_available() and not opt.no_cuda else "cpu")
if torch.cuda.is_available() and opt.no_cuda:
    print("WARNING: You have a CUDA device, so you should probably run without -no_cuda")

opt.model_file = path.join(opt.model_dir, 'policy_networks', 'MPUR-' + opt.policy)

# load the model
model = torch.load(path.join(opt.model_dir, opt.mfile))
if type(model) is dict:
    model = model['model']
model.disable_unet = False
model.opt.lambda_l = opt.lambda_l
model.create_policy_net(opt)
if opt.value_model != '':
    value_function = torch.load(path.join(opt.model_dir, 'value_functions', opt.value_model)).to(opt.device)
    model.value_function = value_function
optimizer = optim.Adam(model.policy_net.parameters(), opt.lrt)  # POLICY optimiser ONLY!
# Have traffic-data point to /misc/vlgscratch4/LecunGroup/nvidia-collab/data/data_i80_v4/
stats = torch.load(f'traffic-data/state-action-cost/data_{opt.dataset}_v{opt.v}/data_stats.pth')
model.stats = stats
if 'ten' in opt.mfile:
    p_z_file = opt.model_dir + opt.mfile + '.pz'
    p_z = torch.load(p_z_file)
    model.p_z = p_z

# Send to GPU if possible
model.to(opt.device)


if 'vae' in opt.mfile:
    opt.model_file += f'-model=vae'
if 'zdropout=0.5' in opt.mfile:
    opt.model_file += '-zdropout=0.5'
elif 'zdropout=0.0' in opt.mfile:
    opt.model_file += '-zdropout=0.0'
if 'model=fwd-cnn-layers' in opt.mfile:
    opt.model_file += '-deterministic'
opt.model_file += f'-nfeature={opt.nfeature}'
opt.model_file += f'-bsize={opt.batch_size}'
opt.model_file += f'-npred={opt.npred}'
opt.model_file += f'-ureg={opt.u_reg}'
opt.model_file += f'-lambdal={opt.lambda_l}'
opt.model_file += f'-lambdaa={opt.lambda_a}'
opt.model_file += f'-gamma={opt.gamma}'
opt.model_file += f'-lrtz={opt.lrt_z}'
opt.model_file += f'-updatez={opt.z_updates}'
opt.model_file += f'-inferz={opt.infer_z}'
opt.model_file += f'-learnedcost={opt.learned_cost}'
opt.model_file += f'-seed={opt.seed}'

if opt.value_model == '':
    opt.model_file += '-novalue'

if opt.learned_cost:
    print('[loading cost regressor]')
    model.cost = torch.load(path.join(opt.model_dir, opt.mfile + '.cost.model'))['model']


print(f'[will save as: {opt.model_file}]')


dataloader = DataLoader(opt, opt.dataset)
model.train()
model.opt.u_hinge = opt.u_hinge
planning.estimate_uncertainty_stats(model, dataloader, n_batches=50, npred=opt.npred)
model.eval()


def train(nbatches, npred):
    model.train()
    model.policy_net.train()

    # tl: target lane
    total_loss_c, total_loss_u, total_loss_l, total_loss_a, total_loss_tl, n_updates, grad_norm = 0, 0, 0, 0, 0, 0, 0
    total_loss_policy = 0
    for j in range(nbatches):
        optimizer.zero_grad()
        inputs, actions, targets, ids, car_sizes = dataloader.get_batch_fm('train', npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        pred, actions, pred_adv = planning.train_policy_net_mpur(
            model, inputs, targets, car_sizes, n_models=10, lrt_z=opt.lrt_z,
            n_updates_z=opt.z_updates, infer_z=(opt.infer_z == 1)
        )
        loss_c = pred[2]  # proximity cost
        loss_l = pred[3]  # lane cost
        loss_u = pred[4]  # uncertainty cost
        loss_a = actions.norm(2, 2).pow(2).mean()  # action regularisation
        # TODO: Calculate target lane loss calculation and add it to line below
        loss_policy = loss_c + opt.u_reg * loss_u + opt.lambda_l * loss_l + opt.lambda_a * loss_a

        if not math.isnan(loss_policy.item()):
            loss_policy.backward()  # back-propagation through time!
            grad_norm += utils.grad_norm(model.policy_net).item()
            torch.nn.utils.clip_grad_norm_(model.policy_net.parameters(), opt.grad_clip)
            optimizer.step()
            total_loss_c += loss_c.item()  # proximity cost
            total_loss_u += loss_u.item()  # uncertainty (reg.)
            total_loss_a += loss_a.item()  # action (reg.)
            total_loss_l += loss_l.item()  # lane cost
            total_loss_policy += loss_policy.item()  # overall total cost
            n_updates += 1
        else:
            print('warning, NaN')  # Oh no... Something got quite fucked up!
            pdb.set_trace()

        if j == 0 and opt.save_movies:
            # save videos of normal and adversarial scenarios
            for b in range(opt.batch_size):
                utils.save_movie(opt.model_file + f'.mov/sampled/mov{b}', pred[0][b], pred[1][b], None, actions[b])
                if pred_adv[0] is not None:
                    utils.save_movie(
                        opt.model_file + f'.mov/adversarial/mov{b}', pred_adv[0][b], pred_adv[1][b], None, actions[b]
                    )

        del inputs, actions, targets, pred

    total_loss_c /= n_updates
    total_loss_u /= n_updates
    total_loss_a /= n_updates
    total_loss_l /= n_updates
    # TODO: Add line for target lane loss
    total_loss_policy /= n_updates
    print(f'[avg grad norm: {grad_norm / n_updates}]')
    # TODO: Add return element for target lane loss
    return total_loss_c, total_loss_l, total_loss_u, total_loss_a, total_loss_policy


def test(nbatches, npred):
    model.train()
    model.policy_net.train()
    total_loss_c, total_loss_u, total_loss_l, total_loss_a, n_updates = 0, 0, 0, 0, 0
    total_loss_policy = 0
    for _ in range(nbatches):
        inputs, actions, targets, ids, car_sizes = dataloader.get_batch_fm('valid', npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        pred, actions, _ = planning.train_policy_net_mpur(model, inputs, targets, car_sizes,
                                                          n_models=10, lrt_z=1.0, n_updates_z=0)
        loss_c = pred[2]
        loss_l = pred[3]
        loss_u = pred[4]
        loss_a = actions.norm(2, 2).pow(2).mean()
        # TODO: Calculate target lane loss calculation and add it to line below
        loss_policy = loss_c + opt.u_reg * loss_u + opt.lambda_l * loss_l + opt.lambda_a * loss_a
        if not math.isnan(loss_policy.item()):
            total_loss_c += loss_c.item()
            total_loss_u += loss_u.item()
            total_loss_a += loss_a.item()
            total_loss_l += loss_l.item()
            total_loss_policy += loss_policy.item()
            n_updates += 1
        else:
            print('warning, NaN')

        del inputs, actions, targets, pred

    total_loss_c /= n_updates
    total_loss_l /= n_updates
    total_loss_u /= n_updates
    total_loss_a /= n_updates
    # TODO: Add line for target lane loss
    total_loss_policy /= n_updates
    # TODO: Add return element for target lane loss
    return total_loss_c, total_loss_l, total_loss_u, total_loss_a, total_loss_policy


print('[training]')
utils.log(opt.model_file + '.log', f'[job name: {opt.model_file}]')
npred = opt.npred if opt.npred != -1 else 16
n_iter = 0

for i in range(500):
    train_losses = train(opt.epoch_size, npred)
    with torch.no_grad():  # Torch, please please please, do not track computations :)
        valid_losses = test(opt.epoch_size // 2, npred)
    n_iter += opt.epoch_size
    model.to('cpu')
    torch.save(dict(
        model=model,
        optimizer=optimizer.state_dict(),
        opt=opt,
        npred=npred,
        n_iter=n_iter,
    ), opt.model_file + '.model')
    if (n_iter / opt.epoch_size) % 10 == 0:
        torch.save(dict(
            model=model,
            optimizer=optimizer.state_dict(),
            opt=opt,
            npred=npred,
            n_iter=n_iter,
        ), opt.model_file + f'step{n_iter}.model')

    model.to(opt.device)
    log_string = f'step {n_iter} | train: [c: {train_losses[0]:.4f}, l: {train_losses[1]:.4f}, ' + \
                 f'u: {train_losses[2]:.4f}, a: {train_losses[3]:.4f}, p: {train_losses[4]:.4f} ] | ' + \
                 f'test: [c: {valid_losses[0]:.4f}, l:{valid_losses[1]:.4f}, u: {valid_losses[2]:.4f}, ' + \
                 f'a: {valid_losses[3]:.4f}, p: {valid_losses[4]:.4f}]'
    print(log_string)
    utils.log(opt.model_file + '.log', log_string)
