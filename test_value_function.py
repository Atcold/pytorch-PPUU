import torch, numpy, argparse, pdb, os, math, time, copy, scipy
import utils
import models
from dataloader import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
plt.ion()

# compare value functions trained on clean data and model predictions. 

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
parser.add_argument('-npred', type=int, default=20)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-gamma', type=float, default=0.99)
parser.add_argument('-dropout', type=float, default=0.0)
parser.add_argument('-nfeature', type=int, default=128)
parser.add_argument('-n_hidden', type=int, default=128)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-epoch_size', type=int, default=2000)
parser.add_argument('-nsync', type=int, default=1)
parser.add_argument('-combine', type=str, default='add')
parser.add_argument('-grad_clip', type=float, default=10)
parser.add_argument('-mfile', type=str, default='model=value-bsize=64-ncond=20-npred=50-lrt=0.0001-nhidden=64-nfeature=64-gclip=10-dropout=0.1-gamma=0.99-nsync=1.model')
parser.add_argument('-fwd_mfile', type=str, default='model=fwd-cnn-ten3-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-nhidden=128-fgeom=1-zeroact=0-zmult=0-dropout=0.1-nz=32-beta=0.0-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model')
parser.add_argument('-mfile2', type=str, default='model=value-bsize=32-ncond=20-npred=50-lrt=0.0001-nhidden=32-nfeature=32-gclip=10-dropout=0.1-gamma=0.99-nsync=1.model')
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

opt.model_file = f'{opt.model_dir}/value_functions/{opt.mfile}'
opt.model_file2 = f'{opt.model_dir}/value_functions_model_outputs/{opt.mfile2}'

print(f'[loading model: {opt.model_file}]')


model = torch.load(opt.model_file)
model.intype('gpu')
model2 = torch.load(opt.model_file2)
model2.intype('gpu')

forward_model = torch.load(opt.model_dir + opt.fwd_mfile)
if type(forward_model) is dict: forward_model = forward_model['model']
forward_model.disable_unet=False
pzfile = opt.model_dir + opt.fwd_mfile + '.pz'
p_z = torch.load(pzfile)
forward_model.p_z = p_z

forward_model.intype('gpu')
# keep dropout
forward_model.train()



model.train()
model2.train()
all_values, all_values2 = [], []
for i in range(50):
    print(i)
    inputs, actions, targets, ids, sizes = dataloader.get_batch_fm('valid')
    inputs = utils.make_variables(inputs)
    targets = utils.make_variables(targets)
    actions = Variable(actions)
#    v = model(inputs[0], inputs[1])

    v = model(targets[0][:, :opt.ncond].contiguous(), targets[1][:, :opt.ncond].contiguous())
    all_values.append(v.data.cpu())

    pred, loss_p = forward_model(inputs, actions, targets, z_dropout=0.0)
    pred_images, pred_states = pred[0], pred[1]
    v_input_images1 = pred_images[:, :opt.ncond].contiguous()
    v_input_states1 = pred_states[:, :opt.ncond].contiguous()
    v2 = model(v_input_images1, v_input_states1)
    all_values2.append(v2.data.cpu())


all_values = torch.stack(all_values).view(-1).cpu().numpy()
all_values2 = torch.stack(all_values2).view(-1).cpu().numpy()


# now look at how the distributions change for real and sampled Z
all_values3, all_values4 = [], []
for i in range(50):
    print(i)
    inputs, actions, targets, ids, sizes = dataloader.get_batch_fm('valid')
    inputs = utils.make_variables(inputs)
    targets = utils.make_variables(targets)
    actions = Variable(actions)
#    v = model(inputs[0], inputs[1])

    v = model(targets[0][:, :opt.ncond].contiguous(), targets[1][:, :opt.ncond].contiguous())
    all_values3.append(v.data.cpu())

    pred, loss_p = forward_model(inputs, actions, targets, z_dropout=0.0, sampling='fp')
    pred_images, pred_states = pred[0], pred[1]
    v_input_images1 = pred_images[:, :opt.ncond].contiguous()
    v_input_states1 = pred_states[:, :opt.ncond].contiguous()
    v2 = model(v_input_images1, v_input_states1)
    all_values4.append(v2.data.cpu())

all_values3 = torch.stack(all_values3).view(-1).cpu().numpy()
all_values4 = torch.stack(all_values4).view(-1).cpu().numpy()

'''
for i in range(100):
    print(i)
    inputs, actions, targets, ids, sizes = dataloader.get_batch_fm('valid')
    inputs = utils.make_variables(inputs)
    targets = utils.make_variables(targets)
    actions = Variable(actions)
    v = model(inputs[0], inputs[1])
    for b in range(opt.batch_size):
        p = int(scipy.stats.percentileofscore(all_values, v[b].item()))
        dirname = f'{opt.model_dir}/viz/{opt.mfile}/ptile-{p}/'
        utils.save_movie(dirname, inputs[0][b], inputs[1][b], None)

'''
    


