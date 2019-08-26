import argparse, pdb, os, pickle, random, sys, numpy
import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from gym.envs.registration import register
import scipy.misc
from dataloader import DataLoader
import utils
from sklearn import decomposition
import sklearn.manifold as manifold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-debug', action='store_true')
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-v', type=int, default=1)
parser.add_argument('-display', type=int, default=0)
parser.add_argument('-seed', type=int, default=9999)
parser.add_argument('-lanes', type=int, default=8)
parser.add_argument('-traffic_rate', type=int, default=15)
parser.add_argument('-n_episodes', type=int, default=1)
parser.add_argument('-ncond', type=int, default=10)
parser.add_argument('-npred', type=int, default=400)
parser.add_argument('-n_batches', type=int, default=200)
parser.add_argument('-n_samples', type=int, default=10)
parser.add_argument('-sampling', type=str, default='fp')
parser.add_argument('-topz_sample', type=int, default=10)
parser.add_argument('-model_dir', type=str, default='models/')
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1-nz=32-beta=0.0-nmix=1-warmstart=1.model')
parser.add_argument('-cuda', type=int, default=1)
parser.add_argument('-save_video', type=int, default=1)
opt = parser.parse_args()

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

opt.save_video = (opt.save_video == 1)
opt.eval_dir = opt.model_dir + f'/eval/'

opt.model_dir += '/'

print(f'[loading {opt.model_dir + opt.mfile}]')
model = torch.load(opt.model_dir + opt.mfile)

model.eval()
if opt.cuda == 1:
    model.intype('gpu')

dataloader = DataLoader(None, opt, opt.dataset)

def compute_pz(nbatches):
    model.p_z = []
    for j in range(nbatches):
        print('[estimating z distribution: {:2.1%}]'.format(float(j)/nbatches), end="\r")
        inputs, actions, targets = dataloader.get_batch_fm('train', opt.npred, (opt.cuda == 1))
        pred, loss_kl = model(inputs, actions, targets, save_z = True)
        del inputs, actions, targets


pzfile = opt.model_dir + opt.mfile + '_100000.pz'
if os.path.isfile(pzfile):
    p_z = torch.load(pzfile)
    graph = torch.load(pzfile + '.graph')
    model.p_z = p_z
    model.knn_indx = graph.get('knn_indx')
    model.knn_dist = graph.get('knn_dist')


model.opt.npred = opt.npred
compute_pz(20)
print('[computing embeddings]')
zpca = decomposition.PCA(n_components=3).fit_transform(torch.cat((model.p_z, p_z), 0))
isomap = manifold.Isomap(n_components=3).fit(model.p_z)
ziso=isomap.fit_transform(model.p_z)

fig = plt.figure(); ax = fig.add_subplot(111, projection='3d'); ax.scatter(zpca[:400, 0], zpca[:400, 1], range(400), s=20, c=range(400), depthshade=True)


'''
pzfile = opt.model_dir + opt.mfile + '.pz'
print(f'[loading p(z) from {pzfile}]')
pz = torch.load(pzfile)
print(f'[loading graph from {pzfile}.graph]')
graph = torch.load(pzfile + '.graph')

dist_null = []
for i in range(100000):
    z1=random.choice(pz)
    z2=random.choice(pz)
    dist_null.append(torch.norm(z1-z2))

dist_true = []
for i in range(model.p_z.size(0)-1):
    z1 = model.p_z[i]
    z2 = model.p_z[i+1]
    dist_true.append(torch.norm(z1-z2))
    
plt.hist(dist_true, bins, alpha=0.5, color='red', normed=True); plt.hist(dist_null, bins, alpha=0.3, color='gray', normed=True); 
plt.legend(['consecutive', 'random'], fontsize=16)
plt.xlabel('L2 distance', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.savefig('plots/distance_histograms.pdf')
#plt.show()
'''
