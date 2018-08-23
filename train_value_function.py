import torch, numpy, argparse, pdb, os, math, time, copy, scipy, os
import utils
import models
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
parser.add_argument('-v', type=int, default=4)
parser.add_argument('-model', type=str, default='policy-cnn-mdn')
parser.add_argument('-layers', type=int, default=3)
parser.add_argument('-fmap_geom', type=int, default=1)
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v8/')
parser.add_argument('-n_episodes', type=int, default=20)
parser.add_argument('-lanes', type=int, default=8)
parser.add_argument('-ncond', type=int, default=20)
parser.add_argument('-npred', type=int, default=50)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-gamma', type=float, default=0.99)
parser.add_argument('-dropout', type=float, default=0.0)
parser.add_argument('-nfeature', type=int, default=128)
parser.add_argument('-n_hidden', type=int, default=128)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-epoch_size', type=int, default=500)
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ten3-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-nhidden=128-fgeom=1-zeroact=0-zmult=0-dropout=0.1-nz=32-beta=0.0-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.model')
parser.add_argument('-nsync', type=int, default=1)
parser.add_argument('-combine', type=str, default='add')
parser.add_argument('-grad_clip', type=float, default=10)
parser.add_argument('-debug', type=int, default=0)
opt = parser.parse_args()


opt.n_actions = 2
opt.n_inputs = opt.ncond
opt.height = 117
opt.width = 24
opt.h_height = 14
opt.h_width = 3
opt.hidden_size = opt.nfeature*opt.h_height*opt.h_width



os.system('mkdir -p ' + opt.model_dir + '/value_functions_model_outputs/')

dataloader = DataLoader(None, opt, opt.dataset)

opt.model_file = f'{opt.model_dir}/value_functions_model_outputs/model=value-bsize={opt.batch_size}-ncond={opt.ncond}-npred={opt.npred}-lrt={opt.lrt}-nhidden={opt.n_hidden}-nfeature={opt.nfeature}-gclip={opt.grad_clip}-dropout={opt.dropout}-gamma={opt.gamma}-nsync={opt.nsync}'

print(f'[will save model as: {opt.model_file}]')

forward_model = torch.load(opt.model_dir + opt.mfile)
if type(forward_model) is dict: forward_model = forward_model['model']
forward_model.disable_unet=True
forward_model.intype('gpu')
# keep dropout
forward_model.train()


model = models.ValueFunction(opt)
model.intype('gpu')
model_ = copy.deepcopy(model)

optimizer = optim.Adam(model.parameters(), opt.lrt)


gamma_mask = Variable(torch.from_numpy(numpy.array([opt.gamma**t for t in range(opt.npred + 1)])).float().cuda()).unsqueeze(0).expand(opt.batch_size, opt.npred + 1)


def train(nbatches):
    model.train()
    total_loss, nb = 0, 0
    all_costs, all_costs2 = [], []
    for i in range(nbatches):
        optimizer.zero_grad()
        inputs, actions, targets, ids, sizes = dataloader.get_batch_fm('train', npred=opt.ncond+opt.npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        # train on noisy model outputs
        pred, loss_p = forward_model(inputs, actions, targets, z_dropout=0.5)
        pred_images, pred_states = pred[0], pred[1]
        v_input_images1 = pred_images[:, :opt.ncond].contiguous()
        v_input_states1 = pred_states[:, :opt.ncond].contiguous()
        v_input_images2 = pred_images[:, -opt.ncond:].contiguous()
        v_input_states2 = pred_states[:, -opt.ncond:].contiguous()
        v = model(v_input_images1, v_input_states1)
        v_ = model(v_input_images2, v_input_states2)
        images, states, _ = targets
        cost, _ = utils.proximity_cost(pred_images[:, opt.ncond:opt.ncond+opt.npred].contiguous(), pred_states[:, opt.ncond:opt.ncond+opt.npred].contiguous(), sizes, unnormalize=True, s_mean=dataloader.s_mean, s_std=dataloader.s_std)

        cost2, _ = utils.proximity_cost(images[:, opt.ncond:opt.ncond+opt.npred].contiguous(), states[:, opt.ncond:opt.ncond+opt.npred].contiguous(), sizes, unnormalize=True, s_mean=dataloader.s_mean, s_std=dataloader.s_std)
        all_costs.append(cost.view(-1).cpu())
        all_costs2.append(cost2.view(-1).cpu())


        v_target = torch.sum(torch.cat((cost, v_), 1) * gamma_mask, 1).view(-1, 1)
        loss = F.mse_loss(v, Variable(v_target.data))
        if not math.isnan(loss.item()):
            loss.backward()
            if opt.grad_clip != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            nb += 1
        else:
            print('warning, NaN')
    return total_loss / nb


def test(nbatches):
    model.train()
    total_loss, nb = 0, 0
    all_values = []
    for i in range(nbatches):
        inputs, actions, targets, ids, sizes = dataloader.get_batch_fm('valid', npred=opt.ncond+opt.npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        # train on noisy model outputs
        pred, loss_p = forward_model(inputs, actions, targets, z_dropout=0.5)
        pred_images, pred_states = pred[0], pred[1]
        v_input_images1 = pred_images[:, :opt.ncond].contiguous()
        v_input_states1 = pred_states[:, :opt.ncond].contiguous()
        v_input_images2 = pred_images[:, -opt.ncond:].contiguous()
        v_input_states2 = pred_states[:, -opt.ncond:].contiguous()

        v = model(v_input_images1, v_input_states1)
        v_ = model(v_input_images2, v_input_states2)
        all_values.append(v.data.cpu())
        cost, _ = utils.proximity_cost(pred_images[:, opt.ncond:].contiguous(), pred_states[:, opt.ncond:].contiguous(), sizes, unnormalize=True, s_mean=dataloader.s_mean, s_std=dataloader.s_std)

        v_target = torch.sum(torch.cat((cost, v_), 1) * gamma_mask, 1).view(-1, 1)
        loss = F.mse_loss(v, Variable(v_target.data))
        if not math.isnan(loss.item()):
            total_loss += loss.item()
            nb += 1
        else:
            print('warning, NaN')
    all_values = torch.stack(all_values).view(-1).cpu().numpy()

    inputs, actions, targets, ids, sizes = dataloader.get_batch_fm('valid', npred=opt.ncond+opt.npred)
    inputs = utils.make_variables(inputs)
    targets = utils.make_variables(targets)
    actions = Variable(actions)
    pred, loss_p = forward_model(inputs, actions, targets, z_dropout=0.5)
    pred_images, pred_states = pred[0], pred[1]
    v_input_images1 = pred_images[:, :opt.ncond].contiguous()
    v_input_states1 = pred_states[:, :opt.ncond].contiguous()
    v = model(v_input_images1, v_input_states1)
    os.system(f'rm -rf {opt.model_dir}/value_functions_model_outputs/viz/{os.path.basename(opt.model_file)}/')
    for b in range(opt.batch_size):
        p = int(scipy.stats.percentileofscore(all_values, v[b].item()))
        dirname = f'{opt.model_dir}/value_functions_model_outputs/viz/{os.path.basename(opt.model_file)}/ptile-{p}/'
        utils.save_movie(dirname, pred_images[b], pred_states[b], None)

    



    return total_loss / nb







print('[training]')
best_valid_loss = 1e6
for i in range(100):
    train_loss = train(opt.epoch_size)
    valid_loss = test(opt.epoch_size)
    
    if opt.nsync == 1:
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model.intype('cpu')
            torch.save(model, opt.model_file + '.model')
            model.intype('gpu')
    else:
        model.intype('cpu')
        torch.save(model, opt.model_file + '.model')
        model.intype('gpu')

    log_string = f'iter {opt.epoch_size*i} | train loss: {train_loss:.5f}, valid: {valid_loss:.5f}, best valid loss: {best_valid_loss:.5f}'
    print(log_string)
    utils.log(opt.model_file + '.log', log_string)
    if opt.nsync > 1 and i % opt.nsync == 0:
        print('[updating target network]')
        utils.log(opt.model_file + '.log', '[updating target network]')
        model_ = copy.deepcopy(model)
