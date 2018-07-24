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
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v6/')
parser.add_argument('-ncond', type=int, default=20)
parser.add_argument('-npred', type=int, default=16)
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-layers', type=int, default=3)
parser.add_argument('-nfeature', type=int, default=256)
parser.add_argument('-n_hidden', type=int, default=256)
parser.add_argument('-beta', type=float, default=0.0, help='weight coefficient of prior loss')
parser.add_argument('-p_dropout', type=float, default=0.0, help='set z=0 with this probability')
parser.add_argument('-nz', type=int, default=2)
parser.add_argument('-n_mixture', type=int, default=10)
parser.add_argument('-context_dim', type=int, default=10)
parser.add_argument('-actions_subsample', type=int, default=4)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-grad_clip', type=float, default=1.0)
parser.add_argument('-epoch_size', type=int, default=500)
parser.add_argument('-curriculum_length', type=int, default=16)
parser.add_argument('-zeroact', type=int, default=0)
parser.add_argument('-warmstart', type=int, default=0)
parser.add_argument('-targetprop', type=int, default=0)
parser.add_argument('-lambda_c', type=float, default=0.0)
parser.add_argument('-lambda_h', type=float, default=0.0)
parser.add_argument('-lambda_lane', type=float, default=0.1)
parser.add_argument('-lrt_traj', type=float, default=0.5)
parser.add_argument('-niter_traj', type=int, default=20)
parser.add_argument('-gamma', type=float, default=1.0)
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ten3-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-nhidden=128-fgeom=1-anoise=0.0-zeroact=0-nz=32-beta=0.0-dropout=0.5-gclip=5.0-warmstart=1.model')
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


os.system('mkdir -p ' + opt.model_dir + '/policy_networks/')


opt.model_file = f'{opt.model_dir}/policy_networks/mbil-nfeature={opt.nfeature}-npred={opt.npred}-lambdac={opt.lambda_c}-lambdah={opt.lambda_h}-lanecost={opt.lambda_lane}-tprop={opt.targetprop}-gamma={opt.gamma}-curr={opt.curriculum_length}-subs={opt.actions_subsample}'

if opt.targetprop == 1:
    opt.model_file += f'-lrt={opt.lrt_traj}-niter={opt.niter_traj}'

opt.model_file += f'-seed={opt.seed}'
print(f'[will save as: {opt.model_file}]')

if os.path.isfile(opt.model_file + '.model'): 
    print('[found previous checkpoint, loading]')
    checkpoint = torch.load(opt.model_file + '.model')
    model = checkpoint['model']
    optimizer1 = optim.Adam(model.policy_net1.parameters(), opt.lrt)
    optimizer2 = optim.Adam(model.policy_net2.parameters(), opt.lrt)
    optimizer1.load_state_dict(checkpoint['optimizer1'])
    optimizer2.load_state_dict(checkpoint['optimizer2'])
    n_iter = checkpoint['n_iter']
    utils.log(opt.model_file + '.log', '[resuming from checkpoint]')
else:
    # load the model
    model = torch.load(opt.model_dir + opt.mfile)['model']
    opt.fmap_geom = 1
    model.create_policy_net(opt)
    model.opt.actions_subsample = opt.actions_subsample
    optimizer1 = optim.Adam(model.policy_net1.parameters(), opt.lrt)
    optimizer2 = optim.Adam(model.policy_net2.parameters(), opt.lrt)
    n_iter = 0

if opt.actions_subsample == -1:
    opt.context_dim = 0

model.intype('gpu')



dataloader = DataLoader(None, opt, opt.dataset)




# training and testing functions. We will compute several losses:
# loss_i: images
# loss_s: states
# loss_c: costs
# loss_p: prior (optional)

def compute_loss(targets, predictions, gamma=1.0, r=True):
    target_images, target_states, target_costs = targets
    pred_images, pred_states, pred_costs, entropy = predictions
    loss_i = F.mse_loss(pred_images, target_images, reduce=False).mean(4).mean(3).mean(2)
    loss_s = F.mse_loss(pred_states, target_states, reduce=False).mean(2)
    loss_c = F.mse_loss(pred_costs, target_costs, reduce=False).mean(2)
    if gamma < 1.0:
        loss_i *= gamma_mask
        loss_s *= gamma_mask
        loss_c *= gamma_mask
    return loss_i.mean(), loss_s.mean(), loss_c.mean(), entropy.mean()

def train(nbatches, npred):
    gamma_mask = torch.Tensor([opt.gamma**t for t in range(npred)]).view(1, -1).cuda()
    model.train()
    model.policy_net1.train()
    model.policy_net2.train()
    total_loss_i, total_loss_s, total_loss_c, total_loss_policy, total_entropy, n_updates = 0, 0, 0, 0, 0, 0
    for i in range(nbatches):
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        inputs, actions, targets = dataloader.get_batch_fm('train', npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        if opt.targetprop == 0:
            pred, _ = model.train_policy_net(inputs, targets)
            loss_i, loss_s, _, entropy = compute_loss(targets, pred)
            proximity_cost = pred[2][:, :, 0]
            lane_cost = pred[2][:, :, 1]
            proximity_cost = proximity_cost * Variable(gamma_mask)
            lane_cost = lane_cost * Variable(gamma_mask)
            loss_c = proximity_cost.mean() + opt.lambda_lane * lane_cost.mean()
            loss_policy = loss_i + loss_s + opt.lambda_c*loss_c - opt.lambda_h*entropy
        else:
            loss_i, loss_s, loss_c, loss_policy = model.train_policy_net_targetprop(inputs, targets, opt)
        if not math.isnan(loss_policy.data[0]):
            loss_policy.backward()
            torch.nn.utils.clip_grad_norm(model.policy_net1.parameters(), opt.grad_clip)
            torch.nn.utils.clip_grad_norm(model.policy_net2.parameters(), opt.grad_clip)
            optimizer1.step()
            optimizer2.step()
            total_loss_i += loss_i.data[0]
            total_loss_s += loss_s.data[0]
            total_loss_c += loss_c.data[0]
            total_entropy += entropy.data[0]
            total_loss_policy += loss_policy.data[0]
            n_updates += 1
        else:
            print('warning, NaN')

        del inputs, actions, targets, pred

    total_loss_i /= n_updates
    total_loss_s /= n_updates
    total_loss_c /= n_updates
    total_loss_policy /= n_updates
    total_entropy /= n_updates
    return total_loss_i, total_loss_s, total_loss_c, total_loss_policy, total_entropy


def test(nbatches, npred):
    gamma_mask = torch.Tensor([opt.gamma**t for t in range(npred)]).view(1, -1).cuda()
    model.eval()
    total_loss_i, total_loss_s, total_loss_c, total_loss_policy, total_entropy, n_updates = 0, 0, 0, 0, 0, 0
    for i in range(nbatches):
        inputs, actions, targets = dataloader.get_batch_fm('valid', npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        pred, pred_actions = model.train_policy_net(inputs, targets, targetprop = opt.targetprop)
        if i == 0 and False:  #TODO
            movie_dir = f'{opt.model_file}.mov/npred{npred}/'
            utils.test_actions(movie_dir, model, inputs, pred_actions, pred)
            
        loss_i, loss_s, _, entropy = compute_loss(targets, pred)
        proximity_cost = pred[2][:, :, 0]
        lane_cost = pred[2][:, :, 1]
        proximity_cost = proximity_cost * Variable(gamma_mask)
        lane_cost = lane_cost * Variable(gamma_mask)
        loss_c = proximity_cost.mean() + opt.lambda_lane * lane_cost.mean()
        loss_policy = loss_i + loss_s + opt.lambda_c*loss_c - opt.lambda_h*entropy
        if not math.isnan(loss_policy.data[0]):
            total_loss_i += loss_i.data[0]
            total_loss_s += loss_s.data[0]
            total_loss_c += loss_c.data[0]
            total_entropy += entropy.data[0]
            total_loss_policy += loss_policy.data[0]
            n_updates += 1
        del inputs, actions, targets, pred

    total_loss_i /= n_updates
    total_loss_s /= n_updates
    total_loss_c /= n_updates
    total_loss_policy /= n_updates
    total_entropy /= n_updates
    return total_loss_i, total_loss_s, total_loss_c, total_loss_policy, total_entropy
        

# set by hand to fit on 12gb GPU
def get_batch_size(npred):
    if npred <= 15:
        return 64
    elif npred <= 50:
        return 32
    elif npred <= 100:
        return 16
    elif npred <= 200:
        return 8
    elif npred <= 400:
        return 4
    elif npred <= 800:
        return 2
    else:
        return 1
    

if opt.test_only == 1:
    print('[testing]')
    valid_losses = test(10, 50)

else:
    print('[training]')
    utils.log(opt.model_file + '.log', f'[job name: {opt.model_file}]')
    npred = opt.npred if opt.npred != -1 else 16
             
    for i in range(500):
        bsize = get_batch_size(npred)
        dataloader.opt.batch_size = bsize
        train_losses = train(opt.epoch_size, npred)
        valid_losses = test(int(opt.epoch_size / 2), npred)
        n_iter += opt.epoch_size
        model.intype('cpu')
        torch.save({'model': model, 
                    'optimizer1': optimizer1.state_dict(),
                    'optimizer2': optimizer2.state_dict(),
                    'opt': opt, 
                    'npred': npred, 
                    'n_iter': n_iter}, 
                   opt.model_file + '.model')
        model.intype('gpu')
        log_string = f'step {n_iter} | npred {npred} | bsize {bsize} | esize {opt.epoch_size} | '
        log_string += utils.format_losses(*train_losses, split='train')
        log_string += utils.format_losses(*valid_losses, split='valid')
        print(log_string)
        utils.log(opt.model_file + '.log', log_string)
        if i > 0 and(i % opt.curriculum_length == 0) and (opt.npred == -1) and npred < 400:
            npred += 8

