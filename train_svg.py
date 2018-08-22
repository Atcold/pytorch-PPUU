import torch, numpy, argparse, pdb, os, time, math, random
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
parser.add_argument('-policy', type=str, default='policy-gauss')
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v8/')
parser.add_argument('-ncond', type=int, default=20)
parser.add_argument('-npred', type=int, default=20)
parser.add_argument('-layers', type=int, default=3)
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-nfeature', type=int, default=256)
parser.add_argument('-n_hidden', type=int, default=256)
parser.add_argument('-dropout', type=float, default=0.0, help='regular dropout')
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-grad_clip', type=float, default=50.0)
parser.add_argument('-epoch_size', type=int, default=500)
parser.add_argument('-lambda_u', type=float, default=0.1)
parser.add_argument('-lambda_a', type=float, default=0.0)
parser.add_argument('-lrt_z', type=float, default=1.0)
parser.add_argument('-z_updates', type=int, default=1)
parser.add_argument('-gamma', type=float, default=0.99)
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ten3-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-nhidden=128-fgeom=1-zeroact=0-zmult=0-dropout=0.1-nz=32-beta=0.0-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model')
parser.add_argument('-value_model', type=str, default='')
parser.add_argument('-load_model_file', type=str, default='')
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

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)



opt.model_file = f'{opt.model_dir}/policy_networks/'
opt.model_file += f'svg-{opt.policy}-nfeature={opt.nfeature}-npred={opt.npred}-lambdau={opt.lambda_u}-lambdaa={opt.lambda_a}-gamma={opt.gamma}-lrtz={opt.lrt_z}-updatez={opt.z_updates}-seed={opt.seed}'
if opt.value_model == '':
    opt.model_file += 'novalue'

print(f'[will save as: {opt.model_file}]')

if os.path.isfile(opt.model_file + '.model') and False: 
    print('[found previous checkpoint, loading]')
    checkpoint = torch.load(opt.model_file + '.model')
    model = checkpoint['model']
    optimizer = optim.Adam(model.policy_net.parameters(), opt.lrt, eps=1e-3)
    optimizer.load_state_dict(checkpoint['optimizer'])
    n_iter = checkpoint['n_iter']
    if opt.test_only == 0:
        utils.log(opt.model_file + '.log', '[resuming from checkpoint]')
else:
    # load the model
    model = torch.load(opt.model_dir + opt.mfile)
    if type(model) is dict: model = model['model']
    model.create_policy_net(opt)
    if opt.value_model != '':
        value_function = torch.load(opt.model_dir + f'/value_functions/{opt.value_model}').cuda()
        model.value_function = value_function
    optimizer = optim.Adam(model.policy_net.parameters(), opt.lrt, eps=1e-3)
    n_iter = 0
    stats = torch.load('/home/mbhenaff/scratch/data/data_i80_v4/data_stats.pth')
    model.stats=stats
    if 'ten' in opt.mfile:
        pzfile = opt.model_dir + opt.mfile + '.pz'
        p_z = torch.load(pzfile)
        model.p_z = p_z



model.intype('gpu')
model.cuda()



dataloader = DataLoader(None, opt, opt.dataset)
model.train()
model.estimate_uncertainty_stats(dataloader, n_batches=100, npred=opt.npred) 
model.eval()



def train(nbatches, npred):
    model.train()
    model.policy_net.train()
    total_loss_c, total_loss_u, total_loss_a, n_updates = 0, 0, 0, 0
    for i in range(nbatches):
        optimizer.zero_grad()
        inputs, actions, targets, ids, car_sizes = dataloader.get_batch_fm('train', npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        pred, actions, pred_adv = model.train_policy_net_svg(inputs, targets, car_sizes, n_models=10, lrt_z=opt.lrt_z, n_updates_z = opt.z_updates)
        loss_c = pred[2]
        loss_u = pred[3]
        loss_a = (actions.norm(2, 2)**2).mean()
        loss_policy = loss_c + opt.lambda_u * loss_u + opt.lambda_a * loss_a
        if not math.isnan(loss_policy.item()):
            loss_policy.backward()
            torch.nn.utils.clip_grad_norm_(model.policy_net.parameters(), opt.grad_clip)
            optimizer.step()
            total_loss_c += loss_c.item()
            total_loss_u += loss_u.item()
            total_loss_a += loss_a.item()
            n_updates += 1
        else:
            print('warning, NaN')
            pdb.set_trace()

        if i == 0: 
            # save videos of normal and adversarial scenarios
            for b in range(opt.batch_size):
                utils.save_movie(opt.model_file + f'.mov/sampled/mov{b}', pred[0][b], pred[1][b], None, actions[b])
                if pred_adv[0] is not None:
                    utils.save_movie(opt.model_file + f'.mov/adversarial/mov{b}', pred_adv[0][b], pred_adv[1][b], None, actions[b])


        del inputs, actions, targets, pred

    total_loss_c /= n_updates
    total_loss_u /= n_updates
    total_loss_a /= n_updates
    return total_loss_c, total_loss_u, total_loss_a

def test(nbatches, npred):
    model.train()
    model.policy_net.train()
    total_loss_c, total_loss_u, total_loss_a, n_updates = 0, 0, 0, 0
    for i in range(nbatches):
        inputs, actions, targets, ids, car_sizes = dataloader.get_batch_fm('valid', npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        pred, actions, _ = model.train_policy_net_svg(inputs, targets, car_sizes, n_models=10, lrt_z=1.0, n_updates_z = 0)
        loss_c = pred[2]
        loss_u = pred[3]
        loss_a = (actions.norm(2, 2)**2).mean()
        loss_policy = loss_c + opt.lambda_u * loss_u
        if not math.isnan(loss_policy.item()):
            total_loss_c += loss_c.item()
            total_loss_u += loss_u.item()
            total_loss_a += loss_a.item()
            n_updates += 1
        else:
            print('warning, NaN')

        del inputs, actions, targets, pred

    total_loss_c /= n_updates
    total_loss_u /= n_updates
    total_loss_a /= n_updates
    return total_loss_c, total_loss_u, total_loss_a






print('[training]')
utils.log(opt.model_file + '.log', f'[job name: {opt.model_file}]')
npred = opt.npred if opt.npred != -1 else 16
             
for i in range(500):
    train_losses = train(opt.epoch_size, npred)
    valid_losses = test(int(opt.epoch_size / 2), npred)
    n_iter += opt.epoch_size
    model.intype('cpu')
    torch.save({'model': model, 
                'optimizer': optimizer.state_dict(),
                'opt': opt, 
                'npred': npred, 
                'n_iter': n_iter}, 
               opt.model_file + '.model')
    model.intype('gpu')
    log_string = f'step {n_iter} | train: [c: {train_losses[0]:.4f}, u: {train_losses[1]:.4f}, a: {train_losses[2]:.4f}] | test: [c: {valid_losses[0]:.4f}, u: {valid_losses[1]:.4f}, a: {valid_losses[2]:.4f}]'
    print(log_string)
    utils.log(opt.model_file + '.log', log_string)

