import argparse
import random
import torch
import numpy
import gym
import pdb
import importlib
import models2 as models
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=0)
# planning params
parser.add_argument('-lrt', type=float, default=0.001)
parser.add_argument('-n_rollouts', type=int, default=10)
parser.add_argument('-npred', type=int, default=20)
parser.add_argument('-n_iter', type=int, default=100)

parser.add_argument('-nb_conditions', type=int, default=10)
parser.add_argument('-nb_predictions', type=int, default=10)
parser.add_argument('-nb_samples', type=int, default=1)
parser.add_argument('-models_dir', type=str, default='./models_il/')
parser.add_argument('-v', type=str, default='3', choices={'3'})
parser.add_argument('-display', type=int, default=1)
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/models/')
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1-nz=64-beta=0.0-nmix=1-warmstart=1.model')

opt = parser.parse_args()

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)

kwargs = {
    'fps': 50,
    'nb_states': opt.nb_conditions,
    'display': opt.display
}

gym.envs.registration.register(
    id='Traffic-v3',
    entry_point='traffic_gym_v3:ControlledI80',
    kwargs=kwargs,
)

print('Building the environment (loading data, if any)')
env = gym.make('Traffic-v' + opt.v)

# load the model
model = torch.load(opt.model_dir + opt.mfile)
model.intype('gpu')
model.stats=torch.load('data_stats.pth')


for episode in range(10):
    env.reset()
    done = False
    action = numpy.zeros((2,))
    images = []
    t = 0
    cntr = 0
    plan = False
    action_seq = None
    while not done:
        observation, reward, done, info = env.step(action)
        if observation is not None:
            t += 1
            if action_seq is None:
                plan = True
            else:
                if cntr >= action_seq.shape[0] or cntr > 5:
                    plan = True
            if plan:
                if len(images) > 0:
                    utils.save_movie('tmp/', torch.from_numpy(numpy.stack(images)))
                action_seq = model.plan_actions_backprop(observation, opt, verbose=True)
                cntr = 0
                plan = False
            action = action_seq[cntr]
            cntr += 1
            print('t={}, cost={}, action=[{}, {}]'.format(t, reward[0][-1], action[0], action[1]))
            images.append(observation[0][-1])
        env.render()
    pdb.set_trace()
    utils.save_movie('tmp/', torch.from_numpy(numpy.stack(images)))
#    utils.save_movie('tmp/', numpy.stack(images))
    

print('Done')
