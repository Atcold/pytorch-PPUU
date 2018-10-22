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

def action_SGD(image, state, dt, cpt):  # with (a,b) being the action
    #scipy.misc.imsave('exp/ex_image_{}.jpg'.format(cpt), np.transpose(image[0].numpy(), (1, 2, 0)))

    speed = torch.norm(state[0, 2:])
    direction = state[0, 2:]/speed

    _, _, car_argmax = proximity_cost(image[:, :].unsqueeze(0), state.unsqueeze(0), return_argmax=True)
    _, _, lane_argmax =  proximity_cost(image[:, :].unsqueeze(0), state.unsqueeze(0), green_channel=0, return_argmax=True)

    a = 0.
    b = 0.
    if car_argmax is not None:
        #a = 50/(np.sign(car_argmax[0])*abs(abs(car_argmax[0]) - 7))
        a = 100/car_argmax[0]
    if lane_argmax is not None:
        #print(lane_argmax)
        if lane_argmax[1] != 0:
            b = 0.001/lane_argmax[1]

    #print("a, b : ", a, b )

    return torch.tensor(a), torch.tensor(b)


total_distance = 0
total_nb_collisions = 0

splits = torch.load('splits.pth')
n_test = len(splits['test_indx'])
for timeslot in [0]:
    for car_id in splits['test_indx'][:20]:
        try:
            observation = env.reset(time_slot=0, vehicle_id=car_id)
        except:
            print("Could not run experiment for car {}. Could not find anything in dataframe.".format(car_id))
            continue
        max_a = 30
        done = False
        a, b = 0., 0.
        cpt = 0


        while not done:
            #a += 1
            #print(f"a = {a}")
            observation, reward, done, info =   env.step(np.array((a,b)))

            if observation is not None:
                input_images, input_states = observation['context'].contiguous(), observation['state'].contiguous()
                speed = input_states[-1:][:, 2:].norm(2, 1)
                #print("speed : ", speed.data)
                #continue
                #input_images[input_images > 100] = 255
                #input_images[input_images <= 100] = 0


                a, b = action_SGD(input_images[-1:], input_states[-1:], opt.delta_t, cpt)
                cpt += 1
                a = torch.max(torch.tensor([a, -speed/opt.delta_t]))
                a = a.clamp(-14, 16)
                if reward['collisions_per_frame'] > 0:
                    break


            env.render()
        total_distance += (info._position - info.look_ahead)[0]
        total_nb_collisions += info.collisions_per_frame
        print('info before', info._position - info.look_ahead)
        print('colisions', info.collisions_per_frame)

        print('Episode completed!')

print("Total MAD : {}".format(float(total_distance)/total_nb_collisions))
print('Done')
