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
parser.add_argument('-seed', type=int, default=333333)
# planning params
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-lrt', type=float, default=0.01)
parser.add_argument('-ncond', type=int, default=10)
parser.add_argument('-npred', type=int, default=50)
parser.add_argument('-nexec', type=int, default=20)
parser.add_argument('-n_rollouts', type=int, default=10)
parser.add_argument('-n_iter', type=int, default=100)
parser.add_argument('-opt_z', type=int, default=0)
parser.add_argument('-opt_a', type=int, default=1)
parser.add_argument('-graph_density', type=float, default=0.001)
parser.add_argument('-models_dir', type=str, default='./models_il/')
parser.add_argument('-v', type=str, default='3', choices={'3'})
parser.add_argument('-display', type=int, default=0)
parser.add_argument('-debug', type=int, default=0)
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v2/')
parser.add_argument('-save_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/planning_results/')
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ten-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-combine=add-nz=32-beta=0.0-dropout=0.5-gclip=1.0-warmstart=1.model')

opt = parser.parse_args()

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)

if opt.dataset == 'i80':
    opt.height = 117
    opt.width = 24
    opt.h_height = 14
    opt.h_width = 3

opt.opt_z = (opt.opt_z == 1)
opt.opt_a = (opt.opt_a == 1)

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

gym.envs.registration.register(
    id='Traffic-v3',
    entry_point='traffic_gym_v3:ControlledI80',
    kwargs={'fps': 10, 'nb_states': opt.ncond, 'display': 0},
)

print('Building the environment (loading data, if any)')
env = gym.make('Traffic-v' + opt.v)


# load the dataset
#dataloader = DataLoader(None, opt, opt.dataset)



total_cost_test = 0
n_batches = 20
plan_file = 'rollouts={}-npred={}-nexec={}-optz={}-opta={}-iter={}-lrt={}'.format(opt.n_rollouts, opt.npred, opt.nexec, opt.opt_z, opt.opt_a, opt.n_iter, opt.lrt)
print('[saving to {}/{}]'.format(opt.save_dir, plan_file))

use_env = True
for j in range(n_batches):
    if use_env:
        print('\n[NEW EPISODE ({})]\n'.format(j))
        movie_dir = '{}/videos_simulator/{}/ep{}/'.format(opt.save_dir, plan_file, j)
        print('[will save to: {}]'.format(movie_dir))
        env.reset()
        inputs = None
        done = False
        images, states, costs = [], [], []        
        while not done:
            if inputs is None:
                print('[finding valid input]')
                while inputs is None:
                    inputs, cost, done, info = env.step(numpy.zeros((2,)))
                print('[done]')
            print('[planning action sequence]')
            a, pred, pred_const, _ = model.plan_actions_backprop(inputs, opt, verbose=True, normalize=True, optimize_z=opt.opt_z, optimize_a=opt.opt_a)        
            cost_test = 0
            print('[executing action sequence]')
            t = 0
            while (t < opt.nexec) and not done:
                inputs, cost, done, info = env.step(a[t])
                images.append(inputs[0][-1])
                states.append(inputs[1][-1])
                costs.append([cost[0][-1], cost[1][-1]])
                t += 1
            costs_ = numpy.stack(costs)
            print(costs_[:, 0].mean() + 0.5*costs_[:, 1].mean())
            if len(images) > 600:
                done = True

        images = numpy.stack(images).transpose(0, 2, 3, 1)
        states = numpy.stack(states)
        costs = numpy.stack(costs)
        cost_test = costs[:, 0].mean() + 0.5*costs[:, 1].mean()
        print(cost_test)
        utils.save_movie('{}/real/'.format(movie_dir), images, states, costs, pytorch=False)
        for i in range(opt.n_rollouts):
            utils.save_movie('{}/imagined/rollout{}/'.format(movie_dir, i), pred[0][i].data, pred[1][i].data, pred[2][i].data, pytorch=True)
    else:
        inputs, actions, targets = dataloader.get_batch_fm('test', opt.npred)
        inputs = utils.make_variables(inputs)
        targets = utils.make_variables(targets)
        actions = utils.Variable(actions)
        a, pred, pred_const, cost_test = model.plan_actions_backprop(inputs, opt, verbose=True, normalize=use_env, optimize_z=opt.opt_z, optimize_a=opt.opt_a)        
        for i in range(4):
            movie_id = j*opt.batch_size + i
            utils.save_movie('tmp_zopt/pred_a_opt/movie{}/'.format(movie_id), pred[0][i].data, pred[1][i].data, pred[2][i].data)
#            utils.save_movie('tmp_zopt/pred_a_const/movie{}/'.format(movie_id), pred_const[0][i].data, pred_const[1][i].data, pred_const[2][i].data)

    total_cost_test += cost_test
    utils.log(opt.save_dir + '/' + plan_file + '.log', 'ep {}, cost: {}'.format(j, cost_test))

total_cost_test /= n_batches
print(total_cost_test)
utils.log(opt.save_dir + '/' + plan_file + '.log', '{}'.format(total_cost_test))
