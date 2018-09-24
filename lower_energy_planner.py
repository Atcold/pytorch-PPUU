import argparse
import numpy as np
import gym
import pdb
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from differentiable_cost import proximity_cost


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

        t = (self.params[0]*self.dt)*direction*self.dt


        trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
        trans[0, 0, 2] = -t[1]/24.
        trans[0, 1, 2] = -t[0]/117.

        grid = affine_grid(trans, torch.Size((1, 1, 117, 24)))

        future_context = grid_sample(image[:, :].float()/255., grid)
        costs, proximity_mask = proximity_cost(future_context.unsqueeze(0), state.unsqueeze(0))

        grid.retain_grad()
        future_context.retain_grad()
        trans.retain_grad()
        t.retain_grad()

        #from ipdb import set_trace; set_trace()
        return costs

model = Model(opt.delta_t)

def action_SGD(image, state, dt, cpt):  # with (a,b) being the action
    #scipy.misc.imsave('exp/ex_image_{}.jpg'.format(cpt), np.transpose(image[0].numpy(), (1, 2, 0)))

    speed = torch.norm(state[0, 2:])
    direction = state[0, 2:]/speed

    #a, b = torch.tensor(0., requires_grad=True), torch.tensor(0., requires_grad=True)
    #optimizer = torch.optim.SGD([model.params], lr=0.01)

    model.init_params()

    for i in range(1):
       #future_image = affine_transformation(image, 0, (10, 10), speed, dt)  # future_image == image when a, b == 0, 0


        costs = model(speed, direction, image, state)

        #if i == 0:
            #print("Proximity cost :  {}".format(costs))

        costs.backward()
        #from ipdb import set_trace; set_trace()

        #grad_a = torch.autograd.grad(loss, a)
        #optimizer.step()
        print(" no flip", model.params.grad.data[0])
        model.params.data[0] = model.params.data[0] - 50000*model.params.grad.data[0]
        #model.params.data[1] = model.params.data[1] - model.params.grad.data[1]
#        model.params.grad.zero_()
        model.params.grad.zero_()

        costs = model(speed, direction, torch.flip(image, [2]), state)
        costs.backward()
        print(" flip", model.params.grad.data[0])
        model.params.data[0] = model.params.data[0] + 50000*model.params.grad.data[0]
        model.params.grad.zero_()


        #print(model.params)
        #a, b -= dc/da, db
    #scipy.misc.imsave('exp/ex_image_affine_{}.jpg'.format(cpt), np.transpose(future_image[0].detach().numpy(), (1, 2, 0)))

    #return model.params.detach()[0], model.params.detach()[1]
    return model.params.detach()

for episode in range(1000):
    env.reset()

    max_a = 30
    done = False
    a, b = 0., 0.
    cpt = 0
    while not done:
        #a += 1
        print(f"a = {a}")
        observation, reward, done, info =   env.step(np.array((a,0)))
        if observation is not None:
            input_images, input_states = observation['context'].contiguous(), observation['state'].contiguous()
            speed = input_states[-1:][:, 2:].norm(2, 1)
            print("speed : ", speed.data)
            #continue
            a, b = action_SGD(input_images[-1:], input_states[-1:], opt.delta_t, cpt)
            cpt += 1
            #a = torch.max(torch.tensor([a, -speed/model.dt]))
            a = a.clamp(-10, 30)


        env.render()

    print('Episode completed!')

print('Done')
