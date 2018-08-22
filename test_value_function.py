import torch, numpy, argparse, pdb, os, math, time, copy
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
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v8/value_functions2/')
parser.add_argument('-n_episodes', type=int, default=20)
parser.add_argument('-lanes', type=int, default=8)
parser.add_argument('-ncond', type=int, default=20)
parser.add_argument('-npred', type=int, default=200)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-gamma', type=float, default=0.99)
parser.add_argument('-dropout', type=float, default=0.0)
parser.add_argument('-nfeature', type=int, default=128)
parser.add_argument('-n_hidden', type=int, default=128)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-epoch_size', type=int, default=2000)
parser.add_argument('-nsync', type=int, default=1)
parser.add_argument('-combine', type=str, default='add')
parser.add_argument('-grad_clip', type=float, default=10)
parser.add_argument('-mfile', type=str, default='model=value-bsize=64-ncond=20-npred=50-lrt=0.0001-nhidden=32-nfeature=32-gclip=10-dropout=0.0-gamma=0.97-nsync=1.model')
parser.add_argument('-debug', type=int, default=0)
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

opt.model_file = f'{opt.model_dir}/{opt.mfile}'

print(f'[loading model: {opt.model_file}]')


model = torch.load(opt.model_file)
model.intype('gpu')


model.eval()
all_values = []
# get quantiles
for i in range(100):
    print(i)
    inputs, actions, targets, ids, sizes = dataloader.get_batch_fm('valid')
    inputs = utils.make_variables(inputs)
    targets = utils.make_variables(targets)
    actions = Variable(actions)
    v = model(inputs[0], inputs[1])
    all_values.append(v.data.cpu())

all_values = torch.stack(all_values).view(-1).cpu().numpy()


for i in range(100):
    print(i)
    inputs, actions, targets, ids, sizes = dataloader.get_batch_fm('valid')
    inputs = utils.make_variables(inputs)
    targets = utils.make_variables(targets)
    actions = Variable(actions)
    v = model(inputs[0], inputs[1])
    for b in range(opt.batch_size):
        p = int(scipy.stats.percentileofscore(all_values, v[b].item()))
        dirname = f'{opt.model_dir}/viz/ptile-{p}/'
        utils.save_movie(dirname, inputs[0][b], inputs[1][b], None)


    


