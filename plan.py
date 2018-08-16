import argparse, os
import random
import torch
import numpy
import gym
import pdb
import importlib
import models
import utils
from dataloader import DataLoader
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-map', type=str, default='i80', choices={'ai', 'i80', 'us101', 'lanker', 'peach'})
parser.add_argument('-seed', type=int, default=333333)
# planning params
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-lrt', type=float, default=0.01)
parser.add_argument('-ncond', type=int, default=20)
parser.add_argument('-npred', type=int, default=1)
parser.add_argument('-nexec', type=int, default=1)
parser.add_argument('-n_rollouts', type=int, default=10)
parser.add_argument('-n_iter', type=int, default=100)
parser.add_argument('-opt_z', type=int, default=0)
parser.add_argument('-opt_a', type=int, default=1)
parser.add_argument('-graph_density', type=float, default=0.001)
parser.add_argument('-display', type=int, default=0)
parser.add_argument('-debug', type=int, default=0)
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v6/policy_networks_il/')
parser.add_argument('-save_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/planning_results_il_v6')
#parser.add_argument('-mfile', type=str, default='mbil-nfeature=256-npred=100-lambdac=0.0-lambdah=0.0-lanecost=0.1-tprop=0-gamma=0.997-curr=10-subs=10-seed=1.model')
parser.add_argument('-mfile', type=str, default='mbil-nfeature=256-npred=100-lambdac=0.0-lambdah=0.0-lanecost=0.1-tprop=0-gamma=0.997-curr=10-subs=10-cdim=2-lossc=1-seed=1.model')

opt = parser.parse_args()

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)

if opt.nexec == -1:
    opt.save_dir += '_single'
else:
    opt.save_dir += '_mpc'
opt.save_dir += '/'

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
    model = torch.load(opt.model_dir + opt.mfile)#['model']
    model.intype('gpu')
    stats = torch.load('/home/mbhenaff/scratch/data/data_i80_v4/data_stats.pth')
    model.stats=stats
    model.policy_net1.stats = stats
    model.policy_net2.stats = stats
    if hasattr(model, 'policy_net'):
        model.policy_net.stats = stats
    if 'ten' in opt.mfile and False:
        pzfile = '/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v{}/'.format(opt.v) + opt.mfile.split('mfile=')[1][:-6] + '_100000.pz'
        p_z = torch.load(pzfile)
        graph = torch.load(pzfile + '.graph')
        model.p_z = p_z
        model.knn_indx = graph.get('knn_indx')
        model.knn_dist = graph.get('knn_dist')
        model.opt.topz_sample = int(model.p_z.size(0)*opt.graph_density)
    return model, stats

model, data_stats = load_model()

gym.envs.registration.register(
    id='I-80-v1',
    entry_point='map_i80_ctrl:ControlledI80',
    kwargs={'fps': 10, 'nb_states': opt.ncond, 'display': 0},
)

print('Building the environment (loading data, if any)')
env_names = {
    'i80': 'I-80-v1',
}

env = gym.make(env_names[opt.map])


total_cost_test = 0
n_batches = 20
if 'mbil' in opt.mfile:
    plan_file = opt.mfile
else:
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
        images, states, costs, actions, mu_list, std_list = [], [], [], [], [], []
        cntr = 0
        actions_context = None
        while not done:
            if inputs is None:
                print('[finding valid input]')
                while inputs is None:
                    inputs, cost, done, info = env.step(numpy.zeros((2,)))
                print('[done]')
            print('[planning action sequence]')
            if 'policy' in opt.mfile:
                _, _, _, a = model(inputs[0].contiguous(), inputs[1].contiguous(), sample=True, unnormalize=True)
            elif 'mbil' in opt.mfile:
                input_images = inputs[0].contiguous()
                input_states = inputs[1].contiguous()
                if cntr % model.opt.actions_subsample == 0:
                    print('new context vector')
                    actions_context, entropy, _, _ = model.policy_net2(input_images, input_states, normalize_inputs=True, normalize_outputs=False)
                a, entropy, mu, std = model.policy_net1(input_images, input_states, context=actions_context, sample=True, normalize_inputs=True, normalize_outputs=True)
                print(entropy.data)
                a = a.cpu().view(1, 2).numpy()
                cntr += 1
            else:
                a, pred, pred_const, _ = model.plan_actions_backprop(inputs, opt, verbose=True, normalize=True, optimize_z=opt.opt_z, optimize_a=opt.opt_a)
            cost_test = 0
            print('[executing action sequence]')
            t = 0
            T = opt.npred if opt.nexec == -1 else opt.nexec
            while (t < T) and not done:
                print(a[t])
                inputs, cost, done, info = env.step(a[t])
                images.append(inputs[0][-1])
                states.append(inputs[1][-1])
                costs.append([cost[0][-1], cost[1][-1]])
#                actions.append(a[t])
                actions.append(((a[t]-data_stats['a_mean'].numpy())/data_stats['a_std']))
                mu_list.append(mu.data.cpu().numpy())
                std_list.append(std.data.cpu().numpy())
                t += 1
            costs_ = numpy.stack(costs)
            print('[true costs: ({:.4f}, {:.4f})]'.format(costs_[:, 0].mean(), costs_[:, 1].mean()))
            if (len(images) > 600) or (opt.nexec == -1):
                done = True

        images = numpy.stack(images).transpose(0, 2, 3, 1)
        states = numpy.stack(states)
        costs = numpy.stack(costs)
        actions = numpy.stack(actions)
        mu_list = numpy.stack(mu_list)
        std_list = numpy.stack(std_list)
        cost_test = costs[:, 0].mean() + 0.5*costs[:, 1].mean()
        utils.save_movie('{}/real/'.format(movie_dir), images, states, costs, actions=actions, mu=mu_list, std=std_list, pytorch=False)
        if 'ten' in opt.mfile and ('mbil' not in opt.mfile):
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
