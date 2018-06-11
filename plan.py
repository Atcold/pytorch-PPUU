import argparse, os
import random
import torch
import numpy
import gym
import pdb
import importlib
import models2 as models
import utils
from dataloader import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-seed', type=int, default=0)
# planning params
parser.add_argument('-batch_size', type=int, default=4)
parser.add_argument('-lrt', type=float, default=0.01)
parser.add_argument('-ncond', type=int, default=10)
parser.add_argument('-npred', type=int, default=200)
parser.add_argument('-n_rollouts', type=int, default=10)
parser.add_argument('-n_iter', type=int, default=100)
parser.add_argument('-graph_density', type=float, default=0.001)
parser.add_argument('-nb_conditions', type=int, default=10)
parser.add_argument('-nb_predictions', type=int, default=10)
parser.add_argument('-nb_samples', type=int, default=1)
parser.add_argument('-models_dir', type=str, default='./models_il/')
parser.add_argument('-v', type=str, default='3', choices={'3'})
parser.add_argument('-display', type=int, default=1)
parser.add_argument('-debug', type=int, default=0)
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v2/')
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ten-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-combine=add-nz=32-beta=0.0-dropout=0.5-gclip=1.0-warmstart=1.model')

opt = parser.parse_args()

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)


# load the model
def load_model():
    importlib.reload(models)
    model = torch.load(opt.model_dir + opt.mfile)
    model.intype('gpu')
    model.stats=torch.load('data_stats.pth')
    pzfile = opt.model_dir + opt.mfile + '_100000.pz'
    p_z = torch.load(pzfile)
    graph = torch.load(pzfile + '.graph')
    model.p_z = p_z
    model.knn_indx = graph.get('knn_indx')
    model.knn_dist = graph.get('knn_dist')
    model.opt.topz_sample = int(model.p_z.size(0)*opt.graph_density)            
    return model

model = load_model()

# load the dataset
dataloader = DataLoader(None, opt, opt.dataset)

for j in range(10):
    print('j')
    inputs, actions, targets = dataloader.get_batch_fm('test', opt.npred)
    inputs = utils.make_variables(inputs)
    targets = utils.make_variables(targets)
    actions = utils.Variable(actions)

    a, pred, pred_const = model.plan_actions_backprop(inputs, opt, verbose=True, normalize=False, optimize_z=True, optimize_a=False)        
    for i in range(4):
        movie_id = j*opt.batch_size + i
        utils.save_movie(f'tmp_zopt/pred_a_opt/movie{movie_id}/', pred[0][i].data, pred[1][i].data, pred[2][i].data)
        utils.save_movie(f'tmp_zopt/pred_a_const/movie{movie_id}/', pred_const[0][i].data, pred_const[1][i].data, pred_const[2][i].data)



