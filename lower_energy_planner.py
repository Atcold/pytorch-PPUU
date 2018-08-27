import argparse
import numpy as np
import gym
import pdb
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-nb_conditions', type=int, default=10)
parser.add_argument('-nb_samples', type=int, default=1)
parser.add_argument('-display', type=int, default=1)
parser.add_argument('-map', type=str, default='i80', choices={'i80', 'us101', 'lanker', 'peach'})

opt = parser.parse_args()

kwargs = {
    'fps': 50,
    'nb_states': opt.nb_conditions,
    'display': opt.display
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

def get_controlled_car(env):
    if env.controlled_car and env.controlled_car['locked']:
        return env.controlled_car['locked']
    else :
        return None

def dE_dxj(env, autonomous_car_id, x, y):
    tot = 0
    energy_vector = np.array([1, 3])

    for car in env.vehicles:
        if car.id is not autonomous_car.id:
            cx, cy = car.get_state().numpy()[:2] + car.get_state().numpy()[2:]*car._dt
            #ener = -100*(1/(car._speed+0.001))*energy_vector[0]*(x - cx)/(energy_vector[0]*(1/(car._speed+0.001))*(cx - x)**2 + energy_vector[1]*(cy - y)**2)**2
            ener = -5000*energy_vector[0]*(x - cx)/(energy_vector[0]*(cx - x)**2 + energy_vector[1]*(cy - y)**2)**2
            tot += ener
    return tot

def aux(env, auto_car_id, x,y):
    tot = 0
    energy_vector = np.array([1, 3])

    for car in env.vehicles:
        if car.id is not auto_car_id:
            cx, cy = car.get_state().numpy()[:2]
            ener = 1./(energy_vector[0]*(cx - x)**2 + energy_vector[1]*(cy - y)**2)**2
            tot += ener
    return tot

import scipy.misc
from torch.nn.functional import affine_grid, grid_sample

def action_SGD(image, cpt):  # with (a,b) being the action
    scipy.misc.imsave('ex_image_{}.jpg'.format(cpt), np.transpose(image[0].numpy(), (1, 2, 0)))
    t_x, t_y = torch.tensor(0., requires_grad=True), torch.tensor(0., requires_grad=True)
    trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], requires_grad=True)
    for i in range(10):
        #future_image = affine_transformation(image, 0, (10, 10), speed, dt)  # future_image == image when a, b == 0, 0
        grid = affine_grid(trans, torch.Size((1, 1, 117, 24)))
        future_image = image
        future_image[0][2] = grid_sample(image[:, 2:3].float(), grid)
        #c = proximity_cost(future_image)
        #print("Proximity cost :  {}".format(c.detach().numpy()[0]))
        #c.backprop()
        #a, b -= dc/da, db
    scipy.misc.imsave('ex_image_affine_{}.jpg'.format(cpt), np.transpose(future_image[0].detach().numpy(), (1, 2, 0)))

    return a, b


def print_energy_fun(time, env, autonomous_car):

    a_x, a_y, _, _ = autonomous_car.get_state().numpy()

    xs = np.arange(0, 1300, 10)
    ys = [aux(env, autonomous_car.id, x, a_y) for x in xs]

    plt.figure(figsize=(30,5))
    plt.plot(xs, ys)
    plt.ylim(0, 0.0004)
    plt.savefig('{}.png'.format(time))



def get_total_energy(env, autonomous_car):
    energy = 0
    for car in env.vehicles:
        if car.id is not autonomous_car.id:
            energy += compute_energy(car.get_state()[:2], autonomous_car.get_state()[:2])

    return energy

for episode in range(1000):
    env.reset()

    max_a = 30
    done = False
    a, b = 0., 0.
    cpt = 0
    while not done:
        observation, reward, done, info =   env.step(np.array((a,b)))
        if observation:
            a, b = action_SGD(observation[0][0:1], cpt)
            cpt += 1

        env.render()

    print('Episode completed!')

print('Done')
