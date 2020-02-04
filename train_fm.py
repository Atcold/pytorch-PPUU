import torch, numpy, argparse, pdb, os, time, math, random
import utils
from dataloader import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import importlib
import models
import torch.nn as nn


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#################################################
# Train an action-conditional forward model
#################################################

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-v', type=int, default=4)
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-model', type=str, default='fwd-cnn')
parser.add_argument('-layers', type=int, default=3, help='layers in frame encoder/decoders')
parser.add_argument('-data_dir', type=str, default='traffic-data/state-action-cost/data_i80_v0/')
parser.add_argument('-model_dir', type=str, default='models')
parser.add_argument('-ncond', type=int, default=20, help='number of conditioning frames')
parser.add_argument('-npred', type=int, default=20, help='number of predictions to make with unrolled fwd model')
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-nfeature', type=int, default=256)
parser.add_argument('-beta', type=float, default=0.0, help='coefficient for KL term in VAE')
parser.add_argument('-ploss', type=str, default='hinge')
parser.add_argument('-z_dropout', type=float, default=0.0, help='set z=0 with this probability')
parser.add_argument('-dropout', type=float, default=0.0, help='regular dropout')
parser.add_argument('-nz', type=int, default=32)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-grad_clip', type=float, default=5.0)
parser.add_argument('-epoch_size', type=int, default=2000)
parser.add_argument('-warmstart', type=int, default=0, help='initialize with pretrained model')
parser.add_argument('-debug', action='store_true')
parser.add_argument('-enable_tensorboard', action='store_true',
                    help='Enables tensorboard logging.')
parser.add_argument('-tensorboard_dir', type=str, default='models',
                    help='path to the directory where to save tensorboard log. If passed empty path' \
                         ' no logs are saved.')
opt = parser.parse_args()

os.system('mkdir -p ' + opt.model_dir)

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
dataloader = DataLoader(None, opt, opt.dataset)


# define model file name
opt.model_file = f'{opt.model_dir}/model={opt.model}-layers={opt.layers}-bsize={opt.batch_size}-ncond={opt.ncond}-npred={opt.npred}-lrt={opt.lrt}-nfeature={opt.nfeature}-dropout={opt.dropout}'

if 'vae' in opt.model:
    opt.model_file += f'-nz={opt.nz}'
    opt.model_file += f'-beta={opt.beta}'
    opt.model_file += f'-zdropout={opt.z_dropout}'

if opt.grad_clip != -1:
    opt.model_file += f'-gclip={opt.grad_clip}'

opt.model_file += f'-warmstart={opt.warmstart}'
opt.model_file += f'-seed={opt.seed}'
print(f'[will save model as: {opt.model_file}]')


# parameters specific to the I-80 dataset
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

# load previous checkpoint or create new model
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
    # specify deterministic model we use to initialize parameters with
    if opt.warmstart == 1:
        prev_model = f'{opt.model_dir}/model=fwd-cnn-layers={opt.layers}-bsize=8-ncond={opt.ncond}-npred={opt.npred}-lrt=0.0001-nfeature={opt.nfeature}-dropout={opt.dropout}-gclip=5.0'
        prev_model += '-warmstart=0-seed=1.step400000.model'
    else:
        prev_model = ''

    if opt.model == 'fwd-cnn':
        # deterministic model
        model = models.FwdCNN(opt, mfile=prev_model)
    elif opt.model == 'fwd-cnn-vae-fp':
        # stochastic VAE model
        model = models.FwdCNN_VAE(opt, mfile=prev_model)
    optimizer = optim.Adam(model.parameters(), opt.lrt)
    n_iter = 0

model.cuda()


# training and testing functions. We will compute several losses:
# loss_i: images
# loss_s: states
# loss_p: relative entropy (optional)

def compute_loss(targets, predictions, reduction='mean'):
    target_images = targets[0]
    target_states = targets[1]
    pred_images, pred_states, _ = predictions
    loss_i = F.mse_loss(pred_images, target_images, reduction=reduction)
    loss_s = F.mse_loss(pred_states, target_states, reduction=reduction)
    return loss_i, loss_s


def expand(x, actions, nrep):
    images, states = x[0], x[1]
    bsize = images.size(0)
    nsteps = images.size(1)
    images_ = images.unsqueeze(0).expand(nrep, bsize, nsteps, 3, opt.height, opt.width)
    images_ = images_.contiguous().view(nrep*bsize, nsteps, 3, opt.height, opt.width)
    states_ = states.unsqueeze(0).expand(nrep, bsize, nsteps, opt.n_inputs)
    states_ = states_.contiguous().view(nrep*bsize, nsteps, opt.n_inputs)
    if actions is not None:
        actions_ = actions.unsqueeze(0).expand(nrep, bsize, nsteps, opt.n_actions)
        actions_ = actions_.contiguous().view(nrep*bsize, nsteps, opt.n_actions).contiguous()
        return [images_, states_, None], actions_
    else:
        return [images_, states_]





def train(nbatches, npred):
    model.train()
    total_loss_i, total_loss_s, total_loss_p = 0, 0, 0
    for i in range(nbatches):
        optimizer.zero_grad()
        inputs, actions, targets, _, _ = dataloader.get_batch_fm('train', npred)
        pred, loss_p = model(inputs[: -1], actions, targets, z_dropout=opt.z_dropout)
        loss_p = loss_p[0]
        loss_i, loss_s = compute_loss(targets, pred)
        loss = loss_i + loss_s + opt.beta*loss_p

        # VAEs get NaN loss sometimes, so check for it
        if not math.isnan(loss.item()):
            loss.backward(retain_graph=False)
            if not math.isnan(utils.grad_norm(model).item()):
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                optimizer.step()

        total_loss_i += loss_i.item()
        total_loss_s += loss_s.item()
        total_loss_p += loss_p.item()
        del inputs, actions, targets

    total_loss_i /= nbatches
    total_loss_s /= nbatches
    total_loss_p /= nbatches
    return total_loss_i, total_loss_s, total_loss_p


def test(nbatches):
    model.eval()
    total_loss_i, total_loss_s, total_loss_p = 0, 0, 0
    for i in range(nbatches):
        inputs, actions, targets, _, _ = dataloader.get_batch_fm('valid')

        pred, loss_p = model(inputs[: -1], actions, targets, z_dropout=opt.z_dropout)
        loss_p = loss_p[0]
        loss_i, loss_s = compute_loss(targets, pred)
        loss = loss_i + loss_s + opt.beta*loss_p

        total_loss_i += loss_i.item()
        total_loss_s += loss_s.item()
        total_loss_p += loss_p.item()
        del inputs, actions, targets

    total_loss_i /= nbatches
    total_loss_s /= nbatches
    total_loss_p /= nbatches
    return total_loss_i, total_loss_s, total_loss_p

writer = utils.create_tensorboard_writer(opt)

print('[training]')
for i in range(200):
    t0 = time.time()
    train_losses = train(opt.epoch_size, opt.npred)
    valid_losses = test(int(opt.epoch_size / 2))

    if writer is not None:
        writer.add_scalar('Loss/train_state_img', train_losses[0], i)
        writer.add_scalar('Loss/train_state_vct', train_losses[1], i)
        writer.add_scalar('Loss/train_relative_entropy', train_losses[2], i)

        writer.add_scalar('Loss/validation_state_img', valid_losses[0], i)
        writer.add_scalar('Loss/validation_state_vct', valid_losses[1], i)
        writer.add_scalar('Loss/validation_relative_entropy', valid_losses[2], i)

    n_iter += opt.epoch_size
    model.cpu()
    torch.save({'model': model,
                'optimizer': optimizer.state_dict(),
                'n_iter': n_iter}, opt.model_file + '.model')
    if (n_iter/opt.epoch_size) % 10 == 0:
        torch.save(model, opt.model_file + f'.step{n_iter}.model')
    model.cuda()
    log_string = f'step {n_iter} | '
    log_string += utils.format_losses(*train_losses, split='train')
    log_string += utils.format_losses(*valid_losses, split='valid')
    print(log_string)
    utils.log(opt.model_file + '.log', log_string)

if writer is not None:
    writer.close()
