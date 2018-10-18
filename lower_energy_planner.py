import argparse
import numpy as np
import gym
import pdb
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from differentiable_cost import proximity_cost
from utils import lane_cost


parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-nb_conditions', type=int, default=10)
parser.add_argument('-nb_samples', type=int, default=1)
parser.add_argument('-display', type=int, default=1)
parser.add_argument('-map', type=str, default='i80', choices={'i80', 'us101', 'lanker', 'peach'})
parser.add_argument('-delta_t', type=float, default=0.1)

opt = parser.parse_args()

kwargs = {
    'fps': 50,
    'nb_states': opt.nb_conditions,
    'display': opt.display,
    'delta_t': opt.delta_t
}

gym.envs.registration.register(
    id='I-80-v1',
    entry_point='map_i80_ctrl:ControlledI80',
    kwargs=kwargs,
)

env_names = {
    'i80': 'I-80-v1',
}

print('Building the environment (loading data, if any)')
env = gym.make(env_names[opt.map])

import scipy.misc
from torch.nn.functional import affine_grid, grid_sample

class Model(torch.nn.Module):
    def __init__(self, dt):
        super(Model, self).__init__()
        self.dt = dt
        self.ortho_dir = torch.zeros([2])
        self.params =  torch.tensor([0., 0], requires_grad=True)


    def init_params(self):
        self.params.data = torch.tensor([0.,0.])


    def forward(self, speed, direction, image, state):

        self.ortho_dir[0] = direction[1]
        self.ortho_dir[1] = -direction[0]

        t = self.params[0]*direction + self.params[1]*self.ortho_dir*self.dt


        trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
        trans[0, 0, 2] = -t[1]/24.
        trans[0, 1, 2] = -t[0]/117.

        grid = affine_grid(trans, torch.Size((1, 1, 117, 24)))

        future_context = grid_sample(image[:, :].float(), grid)
        car_cost, _ = proximity_cost(future_context.unsqueeze(0), state.unsqueeze(0))
        lane_cost, _ =  proximity_cost(future_context.unsqueeze(0), state.unsqueeze(0), green_channel=0)

        return car_cost, lane_cost

model = Model(opt.delta_t)

def action_SGD(image, state, dt, cpt):  # with (a,b) being the action
    #scipy.misc.imsave('exp/ex_image_{}.jpg'.format(cpt), np.transpose(image[0].numpy(), (1, 2, 0)))

    speed = torch.norm(state[0, 2:])
    direction = state[0, 2:]/speed

    _, _, car_argmax = proximity_cost(image[:, :].unsqueeze(0), state.unsqueeze(0), return_argmax=True)
    _, _, lane_argmax =  proximity_cost(image[:, :].unsqueeze(0), state.unsqueeze(0), green_channel=0, return_argmax=True)

    a = 0.
    b = 0.
    if car_argmax is not None:
        print('car_argmax', car_argmax)

        #a = 50/(np.sign(car_argmax[0])*abs(abs(car_argmax[0]) - 7))
        a = 100/car_argmax[0]
    if lane_argmax is not None:
        print(lane_argmax)
        if lane_argmax[1] != 0:
            b = 0.001/lane_argmax[1]

    print("a, b : ", a, b )

    return torch.tensor(a), torch.tensor(b)

for episode in range(1000):
    observation = env.reset(car_id, timeslot=0)

    max_a = 30
    done = False
    a, b = 0., 0.
    cpt = 0
    while not done:
        #a += 1
        print(f"a = {a}")
        observation, reward, done, info =   env.step(np.array((a,b)))
        if observation is not None:
            input_images, input_states = observation['context'].contiguous(), observation['state'].contiguous()
            speed = input_states[-1:][:, 2:].norm(2, 1)
            print("speed : ", speed.data)
            #continue
            #input_images[input_images > 100] = 255
            #input_images[input_images <= 100] = 0


            a, b = action_SGD(input_images[-1:], input_states[-1:], opt.delta_t, cpt)
            cpt += 1
            a = torch.max(torch.tensor([a, -speed/model.dt]))
            a = a.clamp(-14, 16)
            if reward['collisions_per_frame'] > 0:
                break


        env.render()

    print('Episode completed!')

print('Done')
