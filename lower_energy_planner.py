import argparse
import numpy as np
import gym
import pdb
import torch
import utils

from differentiable_cost import proximity_cost
from utils import lane_cost
from dataloader import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-nb_conditions', type=int, default=10)
parser.add_argument('-nb_samples', type=int, default=1)
parser.add_argument('-display', type=int, default=1)
parser.add_argument('-map', type=str, default='i80', choices={'i80', 'us101', 'lanker', 'peach'})
parser.add_argument('-delta_t', type=float, default=0.1)
parser.add_argument('-debug', type=int, default=0)

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

a_mean = np.array([0.24238845705986023, -2.84224752249429e-05])
a_std = np.array([5.077108383178711, 0.002106053987517953])


a_min = (a_mean - 3*a_std)[0]
a_max = (a_mean + 3*a_std)[0]

def action_SGD(image, state, dt, cpt):  # with (a,b) being the action
    speed = torch.norm(state[0, 2:])
    direction = state[0, 2:]/speed

    car_cost, _, car_argmax = proximity_cost(image[:, :].unsqueeze(0), state.unsqueeze(0), return_argmax=True)
    _, _, lane_argmax =  proximity_cost(image[:, :].unsqueeze(0), state.unsqueeze(0), green_channel=0, return_argmax=True)
    
    car_cost = car_cost.detach().numpy()
    
    a = 0.
    b = 0.
    if car_argmax is not None:
        #a = 100/car_argmax[0]
        a = np.sign(car_argmax[0])*car_cost*a_max
    if lane_argmax is not None:
        if lane_argmax[1] != 0:
            b = 0.001/lane_argmax[1]

    return a, b


total_distance = 0
total_nb_collisions = 0

splits = torch.load('../splits.pth')
n_test = len(splits['test_indx'])
dataloader = DataLoader(None, opt, 'i80')


#for j in range(n_test):
for j in range(20):
    car_path = dataloader.ids[splits['test_indx'][j]]
    timeslot, car_id = utils.parse_car_path(car_path)
    print("Starting episode {}/{} with timeslot {}, car_id {}".format(j, n_test, timeslot, car_id))
    observation = env.reset(time_slot=timeslot, vehicle_id=car_id)

    done = False
    a, b = 0., 0.
    cpt = 0


    while True: 
        input_images, input_states = observation['context'].contiguous(), observation['state'].contiguous()
        speed = input_states[-1:][:, 2:].norm(2, 1)

        a, b = action_SGD(input_images[-1:], input_states[-1:], opt.delta_t, cpt)
        cpt += 1
        a = np.amax([a, -speed.numpy()/opt.delta_t])
        a = np.clip(a, a_min, a_max)

        observation, reward, done, info =   env.step(np.array((a,b)))
        if done or reward['collisions_per_frame'] > 0:
            break


        env.render()
    total_distance += (info._position - info.look_ahead)[0]
    total_nb_collisions += info.collisions_per_frame
    print('distance travelled', (info._position - info.look_ahead)[0])
    print('colisions', info.collisions_per_frame)

    print('Episode completed!')

PIXEL_METER_RATIO = 3.7/24
mad = PIXEL_METER_RATIO*float(total_distance)/total_nb_collisions
print("Total MAD : {:2f} m".format(mad))
print('Done')
