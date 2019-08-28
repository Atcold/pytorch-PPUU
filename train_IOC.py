import math
from collections import OrderedDict

import numpy
import os
import pdb
import random
import torch
import torch.optim as optim
from os import path

import planning
import utils
from dataloader import DataLoader
from models import EnergyNet

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#################################################
# Train a policy / controller and cost / energy
#################################################

opt = utils.parse_command_line('IOC')

# Create file_name
opt.model_file = path.join(opt.model_dir, 'policy_networks', 'IOC-' + opt.policy)
utils.build_model_file_name(opt)

os.system('mkdir -p ' + path.join(opt.model_dir, 'policy_networks'))

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)

# Define default device
opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.no_cuda else 'cpu')
if torch.cuda.is_available() and opt.no_cuda:
    print('WARNING: You have a CUDA device, so you should probably run without -no_cuda')

# load the model
model = torch.load(path.join(opt.model_dir, opt.mfile))
if type(model) is dict: model = model['model']
model.opt.lambda_l = opt.lambda_l  # used by planning.py/compute_uncertainty_batch
if opt.value_model != '':
    value_function = torch.load(path.join(opt.model_dir, 'value_functions', opt.value_model)).to(opt.device)
    model.value_function = value_function

# Create policy
model.create_policy_net(opt)
optimizer_policy = optim.Adam(model.policy_net.parameters(), opt.lrt)  # POLICY optimiser ONLY!

# Create EBM
model.energy_net = EnergyNet()
optimizer_energy = optim.Adam(model.energy_net.parameters(), opt.lrt_nrg)  # EBM optimiser ONLY!

# Load normalisation stats
stats = torch.load('traffic-data/state-action-cost/data_i80_v0/data_stats.pth')
model.stats = stats  # used by planning.py/compute_uncertainty_batch

# Send to GPU if possible
model.to(opt.device)

if opt.learned_cost:
    print('[loading cost regressor]')
    model.cost = torch.load(path.join(opt.model_dir, opt.mfile + '.cost.model'))['model']

dataloader = DataLoader(None, opt, opt.dataset)
model.train()
model.opt.u_hinge = opt.u_hinge  # margin for uncertainty penalisation (zero if lower)
planning.estimate_uncertainty_stats(model, dataloader, n_batches=50, npred=opt.npred)
model.eval()


def start(what, nb_batches, npred):
    train = True if what is 'train' else False
    model.train()
    model.policy_net.train()
    n_updates, policy_grad_norm = 0, 0
    grad_norm = dict(
        policy=0,
        energy_net=0,
    )
    total_losses = dict(
        proximity=0,
        uncertainty=0,
        lane=0,
        action=0,
        policy=0,
        cost=0,
        energy=0,
    )
    total_energies = dict(
        expert_state_vct=0,
        expert_state_img=0,
        policy_state_vct=0,
        policy_state_img=0,
    )
    for j in range(nb_batches):
        inputs, expert_actions, targets, ids, car_sizes = dataloader.get_batch_fm(what, npred)
        pred, energies = planning.train_policy_net_ioc(
            model, inputs, targets, car_sizes, expert_actions, infer_z=opt.infer_z
        )
        pred['cost'] = energies['expert_state_vct'].pow(2) + \
                       energies['expert_state_img'].pow(2) + \
                       torch.relu(opt.margin_vct - energies['policy_state_vct']).pow(2) + \
                       torch.relu(opt.margin_img - energies['policy_state_img']).pow(2)
        pred['energy'] = energies['policy_state_vct'].pow(2) + energies['policy_state_img'].pow(2)
        pred['policy'] = pred['proximity'] + \
                         opt.u_reg * pred['uncertainty'] + \
                         opt.lambda_l * pred['lane'] + \
                         opt.lambda_a * pred['action'] + \
                         opt.lambda_e * pred['energy']

        if not math.isnan(pred['policy'].item()):
            if train:
                # Compute gradients for the energy model
                optimizer_energy.zero_grad()
                # Back-propagation though energy net only (no time involved)
                pred['cost'].backward(retain_graph=True)  # don't trash intermediate values just yet
                grad_norm['energy_net'] += utils.grad_norm(model.energy_net).item()
                # TODO: add norm clipping, if energy_net gradients grow too much

                # Compute gradients for the policy
                optimizer_policy.zero_grad()
                pred['policy'].backward()  # back-propagation through time!
                grad_norm['policy'] += utils.grad_norm(model.policy_net).item()
                torch.nn.utils.clip_grad_norm_(model.policy_net.parameters(), opt.grad_clip)

                # Gradient descent for both policy and energy model
                optimizer_policy.step()
                optimizer_energy.step()

            for loss in total_losses: total_losses[loss] += pred[loss].item()
            for energy in total_energies: total_energies[energy] += energies[energy].item()
            n_updates += 1
        else:
            print('warning, NaN')  # Oh no... Something got quite fucked up!
            ipdb.set_trace()

        if j == 0 and opt.save_movies and train:
            # save videos of normal and adversarial scenarios
            for b in range(opt.batch_size):
                state_img = pred['state_img'][b]
                state_vct = pred['state_vct'][b]
                utils.save_movie(opt.model_file + f'.mov/sampled/mov{b}', state_img,
                                 state_vct, None, pred['policy_actions'][b])

        del inputs, targets, pred

    for loss in total_losses: total_losses[loss] /= n_updates
    for energy in total_energies: total_energies[energy] /= n_updates
    if train:
        print(f'[avg grad norm - policy: {grad_norm["policy"] / n_updates:.4f},', end=' ')
        print(f'energy net: {grad_norm["energy_net"] / n_updates:.4f}]')
    return total_losses, total_energies


print('[training]')
utils.log(opt.model_file + '.log', f'[job name: {opt.model_file}]')
nb_iterations = 0
losses = OrderedDict(
    p='proximity',
    l='lane',
    u='uncertainty',
    a='action',
    π='policy',
    c='cost',
    e='energy',
)
energies = OrderedDict(
    ev='expert_state_vct',
    ei='expert_state_img',
    πv='policy_state_vct',
    πi='policy_state_img',
)

for i in range(500):
    train_losses, train_energies = start('train', opt.epoch_size, opt.npred)
    with torch.no_grad():  # Torch, please please please, do not track computations :)
        valid_losses, valid_energies = start('valid', opt.epoch_size // 2, opt.npred)
    nb_iterations += opt.epoch_size
    model.to('cpu')
    save_bundle = dict(
        model=model,
        optimizer_policy=optimizer_policy.state_dict(),
        optimizer_energy=optimizer_energy.state_dict(),
        opt=opt,
        n_iter=nb_iterations,
    )
    torch.save(save_bundle, opt.model_file + '.model')
    if (nb_iterations / opt.epoch_size) % 10 == 0:
        torch.save(save_bundle, opt.model_file + f'step{nb_iterations}.model')

    model.to(opt.device)

    log_string = f'step {nb_iterations} | '
    log_string += 'train: [' + ', '.join(f'{k}: {train_losses[v]:.4f}' for k, v in losses.items())
    log_string += ', '.join(f'{k}: {train_energies[v]:.4f}' for k, v in energies.items()) + '] | '
    log_string += 'valid: [' + ', '.join(f'{k}: {valid_losses[v]:.4f}' for k, v in losses.items())
    log_string += ', '.join(f'{k}: {valid_energies[v]:.4f}' for k, v in energies.items()) + ']'
    print(log_string)
    utils.log(opt.model_file + '.log', log_string)
