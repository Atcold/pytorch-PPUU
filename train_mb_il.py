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
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v4/')
parser.add_argument('-ncond', type=int, default=20)
parser.add_argument('-npred', type=int, default=20)
parser.add_argument('-batch_size', type=int, default=4)
parser.add_argument('-layers', type=int, default=3)
parser.add_argument('-nfeature', type=int, default=256)
parser.add_argument('-n_hidden', type=int, default=128)
parser.add_argument('-beta', type=float, default=0.0, help='weight coefficient of prior loss')
parser.add_argument('-p_dropout', type=float, default=0.0, help='set z=0 with this probability')
parser.add_argument('-nz', type=int, default=2)
parser.add_argument('-n_mixture', type=int, default=10)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-grad_clip', type=float, default=1.0)
parser.add_argument('-epoch_size', type=int, default=4000)
parser.add_argument('-zeroact', type=int, default=0)
parser.add_argument('-warmstart', type=int, default=0)
parser.add_argument('-targetprop', type=int, default=0)
parser.add_argument('-lambda_c', type=float, default=0.0)
parser.add_argument('-lambda_lane', type=float, default=0.1)
parser.add_argument('-lrt_traj', type=float, default=0.5)
parser.add_argument('-niter_traj', type=int, default=20)
parser.add_argument('-gamma', type=float, default=0.98)
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ten-layers=3-bsize=16-ncond=20-npred=20-lrt=0.0001-nhidden=100-nfeature=256-combine=add-nz=32-beta=0.0-dropout=0.5-gclip=1.0-warmstart=0.model')
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


opt.model_file = f'{opt.model_dir}/policy_networks/mbil-nfeature={opt.nfeature}-npred={opt.npred}-mfile={opt.mfile}-lc={opt.lambda_c}-clane={opt.lambda_lane}-tp={opt.targetprop}-g={opt.gamma}'

if opt.targetprop == 1:
    opt.model_file += f'-lrt={opt.lrt_traj}-niter={opt.niter_traj}'

opt.model_file += f'-seed={opt.seed}'
print(f'[will save as: {opt.model_file}]')

if os.path.isfile(opt.model_file + '.model'):
    print('[found previous checkpoint, loading]')
    checkpoint = torch.load(opt.model_file + '.model')
    model = checkpoint
    '''
    model = checkpoint['model']
    optimizer = optim.Adam(model.policy_net.parameters(), opt.lrt)
    optimizer.load_state_dict(checkpoint['optimizer'])
    n_iter = checkpoint['n_iter']
    '''
else:
    # load the model
    model = torch.load(opt.model_dir + opt.mfile)
    opt.fmap_geom = 0
    model.create_policy_net(opt)
    optimizer = optim.Adam(model.policy_net.parameters(), opt.lrt)

model.intype('gpu')



dataloader = DataLoader(None, opt, opt.dataset)

gamma_mask = torch.Tensor([opt.gamma**t for t in range(opt.npred)]).view(1, -1).cuda()



    
# training and testing functions. We will compute several losses:
# loss_i: images
# loss_s: states
# loss_c: costs
# loss_p: prior (optional)

def compute_loss(targets, predictions, gamma=1.0, r=True):
    target_images, target_states, target_costs = targets
    pred_images, pred_states, pred_costs = predictions
    loss_i = F.mse_loss(pred_images, target_images, reduce=False).mean(4).mean(3).mean(2)
    loss_s = F.mse_loss(pred_states, target_states, reduce=False).mean(2)
    loss_c = F.mse_loss(pred_costs, target_costs, reduce=False).mean(2)
    if gamma < 1.0:
        loss_i *= gamma_mask
        loss_s *= gamma_mask
        loss_c *= gamma_mask
    return loss_i.mean(), loss_s.mean(), loss_c.mean()

def train(nbatches, npred):
    model.train()
    model.policy_net.train()
    total_loss_i, total_loss_s, total_loss_c, total_loss_policy, n_updates = 0, 0, 0, 0, 0
    for i in range(nbatches):
        optimizer.zero_grad()
        inputs, actions, targets = dataloader.get_batch_fm('train', npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        if opt.targetprop == 0:
            pred, _ = model.train_policy_net(inputs, targets)
            loss_i, loss_s, _ = compute_loss(targets, pred)
            proximity_cost = pred[2][:, :, 0]
            lane_cost = pred[2][:, :, 1]
            proximity_cost = proximity_cost * Variable(gamma_mask)
            lane_cost = lane_cost * Variable(gamma_mask)
            loss_c = proximity_cost.mean() + opt.lambda_lane * lane_cost.mean()
            loss_policy = loss_i + loss_s + opt.lambda_c*loss_c
        else:
            loss_i, loss_s, loss_c, loss_policy = model.train_policy_net_targetprop(inputs, targets, opt)
        if not math.isnan(loss_policy.data[0]):
            loss_policy.backward()
            torch.nn.utils.clip_grad_norm(model.policy_net.parameters(), opt.grad_clip)
            optimizer.step()
            total_loss_i += loss_i.data[0]
            total_loss_s += loss_s.data[0]
            total_loss_c += loss_c.data[0]
            total_loss_policy += loss_policy.data[0]
            n_updates += 1
        del inputs, actions, targets

    total_loss_i /= n_updates
    total_loss_s /= n_updates
    total_loss_c /= n_updates
    total_loss_policy /= n_updates
    return total_loss_i, total_loss_s, total_loss_c, total_loss_policy


def test(nbatches):
    model.eval()
    total_loss_i, total_loss_s, total_loss_c, total_loss_policy, n_updates = 0, 0, 0, 0, 0
    for i in range(nbatches):
        inputs, actions, targets = dataloader.get_batch_fm('valid')
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        pred, pred_actions = model.train_policy_net(inputs, targets, targetprop = opt.targetprop)
        for b in range(opt.batch_size):
            movie_dir = f'{opt.model_file}.mov/mov{b}/'
            utils.save_movie(movie_dir, pred[0][b].data, pred[1][b].data, pred[2][b].data, pred_actions[b].data)
        pdb.set_trace()
        loss_i, loss_s, _ = compute_loss(targets, pred)
        proximity_cost = pred[2][:, :, 0]
        lane_cost = pred[2][:, :, 1]
        proximity_cost = proximity_cost * Variable(gamma_mask)
        lane_cost = lane_cost * Variable(gamma_mask)
        loss_c = proximity_cost.mean() + opt.lambda_lane * lane_cost.mean()
        loss_policy = loss_i + loss_s + opt.lambda_c*loss_c
        if not math.isnan(loss_policy.data[0]):
            total_loss_i += loss_i.data[0]
            total_loss_s += loss_s.data[0]
            total_loss_c += loss_c.data[0]
            total_loss_policy += loss_policy.data[0]
        del inputs, actions, targets

    total_loss_i /= n_updates
    total_loss_s /= n_updates
    total_loss_c /= n_updates
    total_loss_policy /= n_updates
    return total_loss_i, total_loss_s, total_loss_c, total_loss_policy
        
if opt.test_only == 1:
    print('[testing]')
    valid_losses = test(10)

else:
    print('[training]')
    for i in range(500):
        train_losses = train(opt.epoch_size, opt.npred)
        valid_losses = test(int(opt.epoch_size / 2))
        n_iter += opt.epoch_size
        model.intype('cpu')
        torch.save({'model': model, 
                    'optimizer': optimizer.state_dict(),
                    'n_iter': (i+1)*opt.epoch_size}, 
                   opt.model_file + '.model')
        model.intype('gpu')
        log_string = f'step {n_iter} | '
        log_string += utils.format_losses(*train_losses, 'train')
        log_string += utils.format_losses(*valid_losses, 'valid')
        print(log_string)
        utils.log(opt.model_file + '.log', log_string)
