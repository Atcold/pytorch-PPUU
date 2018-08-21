import torch, numpy, argparse, pdb, os, time, math, random
import utils
from dataloader import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import importlib
import models
import torch.nn as nn

#################################################
# Train an action-conditional forward model
#################################################

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-v', type=int, default=4)
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-model', type=str, default='fwd-cnn3')
parser.add_argument('-layers', type=int, default=3)
parser.add_argument('-data_dir', type=str, default='traffic-data/state-action-cost/data_i80_v0/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v8/')
parser.add_argument('-ncond', type=int, default=20)
parser.add_argument('-npred', type=int, default=20)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-nfeature', type=int, default=256)
parser.add_argument('-n_hidden', type=int, default=128)
parser.add_argument('-fmap_geom', type=int, default=1)
parser.add_argument('-sigmoid_out', type=int, default=1)
parser.add_argument('-beta', type=float, default=0.0, help='weight coefficient of prior loss')
parser.add_argument('-a_noise_penalty', type=float, default=0.0)
parser.add_argument('-ploss', type=str, default='hinge')
parser.add_argument('-z_dropout', type=float, default=0.0, help='set z=0 with this probability')
parser.add_argument('-dropout', type=float, default=0.0, help='regular dropout')
parser.add_argument('-nz', type=int, default=32)
parser.add_argument('-n_mixture', type=int, default=10)
parser.add_argument('-z_sphere', type=int, default=0)
parser.add_argument('-adv_loss', type=float, default=0.0)
parser.add_argument('-action_indep_net', type=int, default=0)
parser.add_argument('-a_indep_lambda', type=float, default=0.0)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-lrt_d', type=float, default=0.0001)
parser.add_argument('-grad_clip', type=float, default=5.0)
parser.add_argument('-epoch_size', type=int, default=2000)
parser.add_argument('-zeroact', type=int, default=0)
parser.add_argument('-warmstart', type=int, default=0)
parser.add_argument('-combine', type=str, default='add')
parser.add_argument('-zmult', type=int, default=0)
parser.add_argument('-debug', type=int, default=0)
opt = parser.parse_args()

os.system('mkdir -p ' + opt.model_dir)

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
dataloader = DataLoader(None, opt, opt.dataset)


# define model file name
opt.model_file = f'{opt.model_dir}/model={opt.model}-layers={opt.layers}-bsize={opt.batch_size}-ncond={opt.ncond}-npred={opt.npred}-lrt={opt.lrt}-nfeature={opt.nfeature}-nhidden={opt.n_hidden}-fgeom={opt.fmap_geom}-zeroact={opt.zeroact}-zmult={opt.zmult}-dropout={opt.dropout}'

if ('vae' in opt.model) or ('fwd-cnn-ten' in opt.model):
    opt.model_file += f'-nz={opt.nz}'
    opt.model_file += f'-beta={opt.beta}'
    opt.model_file += f'-zdropout={opt.z_dropout}'

if ('fwd-cnn-ten' in opt.model) and opt.beta > 0:
    if opt.ploss == 'pdf':
        opt.model_file += f'-nmix={opt.n_mixture}'
    elif opt.ploss == 'hinge':
        opt.z_sphere = 1
        opt.model_file += f'-ploss=hinge'
        opt.model_file += f'-zsphere={opt.z_sphere}'


if opt.grad_clip != -1:
    opt.model_file += f'-gclip={opt.grad_clip}'

if opt.adv_loss > 0.0:
    opt.model_file += f'-advloss={opt.adv_loss}-lrtd={opt.lrt_d}'

if opt.action_indep_net == 1:
    opt.model_file += f'-aindep={opt.a_indep_lambda}'

opt.model_file += f'-warmstart={opt.warmstart}'
opt.model_file += f'-seed={opt.seed}'
print(f'[will save model as: {opt.model_file}]')


# specific to the I-80 dataset
opt.n_inputs = 4
opt.n_actions = 2
opt.height = 117
opt.width = 24
if opt.layers == 3:
    opt.h_height = 14
    opt.h_width = 3
elif opt.layers == 4:
    opt.h_height = 7
    opt.h_width = 1
opt.hidden_size = opt.nfeature*opt.h_height*opt.h_width

mfile = opt.model_file + '.model'
if os.path.isfile(mfile):
    print(f'[loading previous checkpoint: {mfile}]')
    checkpoint = torch.load(mfile)
    model = checkpoint['model']
    model.cuda()
    optimizer = optim.Adam(model.parameters(), opt.lrt)
    optimizer.load_state_dict(checkpoint['optimizer'])
    n_iter = checkpoint['n_iter']
    utils.log(opt.model_file + '.log', '[resuming from checkpoint]')
else:
    # create new model
    # specify deterministic model we use to initialize parameters with
    if opt.warmstart == 1:
        prev_model = f'{opt.model_dir}/model=fwd-cnn3-layers=3-bsize=64-ncond={opt.ncond}-npred={opt.npred}-lrt=0.0001-nfeature={opt.nfeature}-nhidden=128-fgeom={opt.fmap_geom}-zeroact={opt.zeroact}-zmult={opt.zmult}-dropout={opt.dropout}-gclip=5.0-warmstart=0-seed=1.model'
    else:
        prev_model = ''

    if opt.model == 'fwd-cnn':
        model = models.FwdCNN(opt, mfile=prev_model)
    elif opt.model == 'fwd-cnn2':
        model = models.FwdCNN2(opt, mfile=prev_model)
    elif opt.model == 'fwd-cnn3':
        model = models.FwdCNN3(opt, mfile=prev_model)
    elif opt.model == 'fwd-cnn3-stn':
        model = models.FwdCNN3_STN(opt, mfile=prev_model)
    elif opt.model == 'fwd-cnn-ten':
        model = models.FwdCNN_TEN(opt, mfile=prev_model)
    elif opt.model == 'fwd-cnn-ten2':
        model = models.FwdCNN_TEN2(opt, mfile=prev_model)
    elif opt.model == 'fwd-cnn-ten3':
        model = models.FwdCNN_TEN3(opt, mfile=prev_model)
    elif opt.model == 'fwd-cnn-vae3-fp':
        model = models.FwdCNN_VAE3(opt, mfile=prev_model)
    elif opt.model == 'fwd-cnn-vae3-lp':
        model = models.FwdCNN_VAE3(opt, mfile=prev_model)
    elif opt.model == 'fwd-cnn-vae-fp':
        model = models.FwdCNN_VAE_FP(opt, mfile=prev_model)
    elif opt.model == 'fwd-cnn-vae-lp':
        model = models.FwdCNN_VAE_LP(opt, mfile=prev_model)
    optimizer = optim.Adam(model.parameters(), opt.lrt, epsilon=1e-3)
    n_iter = 0

model.intype('gpu')

if opt.adv_loss > 0:
    discriminator = nn.Sequential(
        nn.Linear(opt.n_actions + opt.nz, opt.n_hidden),
        nn.ReLU(),
        nn.Linear(opt.n_hidden, 1),
        nn.Sigmoid()
        )
    discriminator.cuda()
    optimizer_d = optim.Adam(discriminator.parameters(), opt.lrt_d)

# training and testing functions. We will compute several losses:
# loss_i: images
# loss_s: states
# loss_c: costs
# loss_p: prior (optional)

def compute_loss(targets, predictions, r=True):
    target_images = targets[0]
    target_states = targets[1]
    target_costs = targets[2]
    pred_images, pred_states, pred_costs, _ = predictions
    loss_i = F.mse_loss(pred_images, target_images, reduce=r)
    loss_s = F.mse_loss(pred_states, target_states, reduce=r)
    loss_c = F.mse_loss(pred_costs, target_costs, reduce=r)
    if not r:
        loss_i = loss_i.mean(4).mean(3).mean(2)
        loss_s = loss_s.mean(2)
        loss_c = loss_c.mean(2)
    return loss_i, loss_s, loss_c


def discriminator_loss(actions, z):
    inputs = torch.cat((actions, z), 2)
    out = discriminator(inputs.view(-1, opt.n_actions + opt.nz))
    targets = torch.cuda.FloatTensor(out.size()).fill_(0.5)
    loss_d = F.binary_cross_entropy(out, Variable(targets))
    return loss_d


def perturbation_loss(inputs, actions, targets, pred):
    bsize = actions.size(0)
    npred = actions.size(1)
    noise = Variable(torch.randn(actions.size()).cuda())
    pred_n, _ = model(inputs, actions + noise, targets, z_dropout=opt.z_dropout)
    loss_i_n, loss_s_n, loss_c_n = compute_loss(pred, pred_n, r=False)
    pred_i = pred[0].view(bsize, npred, -1)
    pred_s = pred[1].view(bsize, npred, -1)
    pred_n_i = pred_n[0].view(bsize, npred, -1)
    pred_n_s = pred_n[1].view(bsize, npred, -1)
    noise_norm = noise.norm(2, 2)
    pred_i_norm = pred_i.norm(2, 2)
    pred_s_norm = pred_s.norm(2, 2)
    pred_n_i_norm = pred_n_i.norm(2, 2)
    pred_n_s_norm = pred_n_s.norm(2, 2)
    loss_n = (1.0 - loss_i_n / (pred_i_norm + pred_n_i_norm))
    loss_n *= noise_norm
#    loss_n = -(loss_i_n/(pred_n_i.norm()+pred_i.norm()))
    return loss_n.mean()



def train_discriminator(actions, z):
    discriminator.zero_grad()
    optimizer_d.zero_grad()
    actions = actions.clone()
    bsize = actions.size(0)
    half = int(bsize/2)
    perm = torch.randperm(half).cuda()
    actions[:half] = actions[perm]
    inputs = torch.cat((actions, z), 2)
    targets = torch.cuda.FloatTensor(bsize, opt.npred)
    targets[:half].fill_(1)
    targets[half:].fill_(0)
    targets = targets.view(-1, 1)
    out = discriminator(inputs.view(-1, opt.n_actions + opt.nz))
    loss_d = F.binary_cross_entropy(out, Variable(targets))
    loss_d.backward()
    optimizer_d.step()
    acc = (out.data[:(half*opt.npred)].gt(0.5).sum() + out.data[(half*opt.npred):].lt(0.5).sum())/(bsize*opt.npred)
    return loss_d, acc


def train(nbatches, npred):
    model.train()
    total_loss_i, total_loss_s, total_loss_c, total_loss_p, total_loss_p2 = 0, 0, 0, 0, 0
    for i in range(nbatches):
        optimizer.zero_grad()
        inputs, actions, targets = dataloader.get_batch_fm('train', npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        if opt.zeroact == 1:
            actions.data.zero_()
        pred, loss_p = model(inputs, actions, targets, z_dropout=opt.z_dropout)
        loss_i, loss_s, loss_c = compute_loss(targets, pred)
        loss = loss_i + loss_s + loss_c + opt.beta*loss_p[0]

        if not math.isnan(loss.item()):
            loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

        if opt.adv_loss > 0.0:
            z = pred[3]
            z.detach()
            loss_d, acc = train_discriminator(actions, z)

        total_loss_i += loss_i.item()
        total_loss_s += loss_s.item()
        total_loss_c += loss_c.item()
        total_loss_p += loss_p[0].item()
        if len(loss_p) > 1:
            total_loss_p2 += loss_p[1].item()
        del inputs, actions, targets

    total_loss_i /= nbatches
    total_loss_s /= nbatches
    total_loss_c /= nbatches
    total_loss_p /= nbatches
    total_loss_p2 /= nbatches
    return total_loss_i, total_loss_s, total_loss_c, total_loss_p, total_loss_p2


def test(nbatches):
    model.eval()
    total_loss_i, total_loss_s, total_loss_c, total_loss_p, total_loss_p2 = 0, 0, 0, 0, 0
    for i in range(nbatches):
        inputs, actions, targets = dataloader.get_batch_fm('valid')
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = Variable(actions)
        if opt.zeroact == 1:
            actions.data.zero_()
        pred, loss_p = model(inputs, actions, targets, z_dropout=0)
        loss_i, loss_s, loss_c = compute_loss(targets, pred)

        total_loss_i += loss_i.item()
        total_loss_s += loss_s.item()
        total_loss_c += loss_c.item()
        total_loss_p += loss_p[0].item()
        if len(loss_p) > 1:
            total_loss_p2 += loss_p[1].item()

        if i == 0 and opt.model_dir != 'tmp/' and False: #TODO
            utils.test_actions(opt.model_file + '.mov/', model, inputs, actions, targets)

        del inputs, actions, targets

    total_loss_i /= nbatches
    total_loss_s /= nbatches
    total_loss_c /= nbatches
    total_loss_p /= nbatches
    total_loss_p2 /= nbatches
    return total_loss_i, total_loss_s, total_loss_c, total_loss_p, total_loss_p2


print('[training]')
for i in range(200):
    t0 = time.time()
    train_losses = train(opt.epoch_size, opt.npred)
    valid_losses = test(int(opt.epoch_size / 2))
    n_iter += opt.epoch_size
    model.intype('cpu')
    torch.save({'model': model,
                'optimizer': optimizer.state_dict(),
                'n_iter': n_iter}, opt.model_file + '.model')
    if (n_iter/opt.epoch_size) % 10 == 0:
        torch.save(model, opt.model_file + f'.step{n_iter}.model')
    model.intype('gpu')
    log_string = f'step {n_iter} | '
    log_string += utils.format_losses(*train_losses, split='train')
    log_string += utils.format_losses(*valid_losses, split='valid')
    print(log_string)
    utils.log(opt.model_file + '.log', log_string)

