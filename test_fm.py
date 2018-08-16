import argparse, pdb, os
import gym
import numpy as np
import os
import pickle
import random
import torch
from torch.autograd import Variable
from gym.envs.registration import register
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument('-map', type=str, default='i80', choices={'ai', 'i80', 'us101', 'lanker', 'peach'})
parser.add_argument('-display', type=int, default=1)
parser.add_argument('-seed', type=int, default=9999)
parser.add_argument('-lanes', type=int, default=8)
parser.add_argument('-traffic_rate', type=int, default=15)
parser.add_argument('-n_episodes', type=int, default=1)
parser.add_argument('-ncond', type=int, default=10)
parser.add_argument('-npred', type=int, default=100)
parser.add_argument('-tie_action', type=int, default=0)
parser.add_argument('-sigmout', type=int, default=1)
parser.add_argument('-n_samples', type=int, default=5)
opt = parser.parse_args()

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

tags = {'wrapper_config.TimeLimit.max_episodesteps': 100}
kwargs = {
    'display': opt.display,
    'nb_lanes': opt.lanes,
    'traffic_rate': opt.traffic_rate,
    'state_image': 1,
    'store': 1,
}

register(
    id='Traffic-v0',
    entry_point='traffic_gym:Simulator',
    kwargs=kwargs
)

register(
    id='I-80-v0',
    entry_point='map_i80:I80',
    kwargs=kwargs
)

gym.envs.registration.register(
    id='US-101-v0',
    entry_point='map_us101:US101',
    kwargs=kwargs,
)

gym.envs.registration.register(
    id='Lankershim-v0',
    entry_point='map_lanker:Lankershim',
    kwargs=kwargs,
)

gym.envs.registration.register(
    id='Peachtree-v0',
    entry_point='map_peach:Peachtree',
    kwargs=kwargs,
)

env_names = {
    'ai': 'Traffic-v0',
    'i80': 'I-80-v0',
    'us101': 'US-101-v0',
    'lanker': 'Lankershim-v0',
    'peach': 'Peachtree-v0',
}

env = gym.make(env_names[opt.map])


mdir = 'dataset_i80/'
#opt.mfile = 'model=fwd-cnn-een-fp-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-sigmout=1-tieact=0-nz=8.model'
opt.mfile = 'model=fwd-cnn-bsize=32-ncond=10-npred=30-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-warmstart=0.model'
model = torch.load(mdir + '/models/' + opt.mfile)
model.opt.tie_action = 0
model.opt.npred = opt.npred


def run_episode():
    states_, actions_, rewards_ = [], [], []
    done = False

    state, objects = env.reset()
    for t in range(2000):
        if t > 500:
            vid = 0
            vehicles = env.vehicles
            for v in vehicles:
                print('here')
                images = torch.stack(v._states_image).permute(0, 3, 2, 1).float()
                images /= 255.0
                images = Variable(images[-opt.ncond:].clone().float().unsqueeze(0))

                dirname = f'{mdir}/videos/{opt.mfile}'
                actions = torch.zeros(1, opt.npred, 3)
                actions[:, :, 2].fill_(0)
                actions[:, :, 1].fill_(+0.0)
                for s in range(opt.n_samples):
                    pred, _ = model(images, Variable(actions), None)
                    print(pred.norm())
                    pred = pred.squeeze().permute(0, 2, 3, 1).data.numpy()
                    dirname_movie = '{}/pred{:d}/action0/sample{:d}/'.format(dirname, vid, s)
                    os.system("mkdir -p " + dirname_movie)
                    for t in range(opt.npred):
                        if t > 0:
                            p = (pred[t] + pred[t-1])/2
                        else:
                            p = pred[t]
                        scipy.misc.imsave(dirname_movie + '/im{:05d}.png'.format(t), p)
                vid += 1

        else:
            action = None
        state, reward, vehicles = env.step(action)
        env.render()


for i in range(opt.n_episodes):
    print(f'episode {i + 1}/{opt.n_episodes}')
    runs = run_episode()
