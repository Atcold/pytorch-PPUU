import torch, numpy, argparse, pdb, os
import utils
import models2 as models
from dataloader import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


###########################################
# Train an imitation learner model
###########################################

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-model', type=str, default='policy-cnn-mdn')
parser.add_argument('-nshards', type=int, default=40)
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models/policy/')
parser.add_argument('-n_episodes', type=int, default=20)
parser.add_argument('-lanes', type=int, default=8)
parser.add_argument('-ncond', type=int, default=10)
parser.add_argument('-npred', type=int, default=1)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-nfeature', type=int, default=128)
parser.add_argument('-n_hidden', type=int, default=100)
parser.add_argument('-n_mixture', type=int, default=20)
parser.add_argument('-nz', type=int, default=2)
parser.add_argument('-beta', type=float, default=0.1)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-warmstart', type=int, default=0)
parser.add_argument('-epoch_size', type=int, default=1000)
parser.add_argument('-combine', type=str, default='add')
parser.add_argument('-grad_clip', type=float, default=10)
parser.add_argument('-debug', type=int, default=0)
opt = parser.parse_args()


opt.n_actions = 2
opt.n_inputs = opt.ncond
if opt.dataset == 'simulator':
    opt.height = 97
    opt.width = 20
    opt.h_height = 12
    opt.h_width = 2
    opt.model_dir += f'_{opt.nshards}-shards/'

elif opt.dataset == 'i80':
    opt.height = 117
    opt.width = 24
    opt.h_height = 14
    opt.h_width = 3
    opt.hidden_size = opt.nfeature*opt.h_height*opt.h_width



if opt.dataset == 'simulator':
    opt.model_dir += f'_{opt.nshards}-shards/'
    data_file = f'{opt.data_dir}/traffic_data_lanes={opt.lanes}-episodes=*-seed=*.pkl'
else:
    data_file = None
os.system('mkdir -p ' + opt.model_dir)

dataloader = DataLoader(data_file, opt, opt.dataset)

npred = 20
#mfile = 'model=policy-cnn-mdn-bsize=32-ncond=10-npred=5-lrt=0.0001-nhidden=100-nfeature=128-nmixture=10-gclip=10.model'
#mfile = 'model=policy-cnn-mdn-bsize=32-ncond=10-npred=10-lrt=0.0001-nhidden=100-nfeature=128-nmixture=10-gclip=10.model'
mfile = 'model=policy-cnn-mdn-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-nmixture=10-gclip=10.model'
policy = torch.load('/home/mbhenaff/scratch/models/policy/' + mfile)


inputs, actions, targets = dataloader.get_batch_fm('test')
inputs = utils.make_variables(inputs)
targets = utils.make_variables(targets)
actions = Variable(actions)
pi, mu, sigma, _ = policy(inputs[0].cpu(), inputs[1].cpu(), unnormalize=False)

mu = mu.view(opt.batch_size, -1, npred, 2)
sigma = sigma.view(opt.batch_size, -1, npred, 2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
s=1
for i in range(10):
    xs = mu[s][i][:, 0].data.cpu().numpy()
    ys = mu[s][i][:, 1].data.cpu().numpy()
    zs = numpy.array(range(npred)).astype(float)
    ax.scatter(xs, ys, zs, zdir='z', s=20)
ax.set_xlabel('acceleration')
ax.set_ylabel('angle')
ax.set_zlabel('time')

plt.show()



