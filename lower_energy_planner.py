import argparse
import numpy as np
import gym
import pdb
import torch
import utils
import progressbar

from differentiable_cost import proximity_cost
from utils import lane_cost
from dataloader import DataLoader

PIXEL_METER_RATIO = 3.7/24

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


a_min = (a_mean - 5*a_std)[0]
a_max = (a_mean + 3*a_std)[0]

def action_SGD(image, state, force_cst, car_length):  # with (a,b) being the action
    SCALE = 0.25
    safe_factor = 1.5

    speed = state[:, 2:].norm(2, 1) * SCALE #pixel/s

    safe_distance = torch.abs(speed) * safe_factor + (1*24/3.7) * SCALE  # plus one metre (TODO change)
    safe_distance = safe_distance.numpy()

    direction = state[0, 2:]/speed

    car_cost, _, car_argmax = proximity_cost(image[:, :].unsqueeze(0), state.unsqueeze(0), return_argmax=True)
    _, _, lane_argmax =  proximity_cost(image[:, :].unsqueeze(0), state.unsqueeze(0), green_channel=0, return_argmax=True)
    
    car_cost = car_cost.detach().numpy()
    
    a = 0.
    b = 0.
    if car_argmax is not None:
        distance = car_argmax[0]
        a = np.sign(distance)*force_cst*a_std*(1 - abs(distance))/safe_distance
        #a = np.sign(distance)*force_cst*a_std[0]*(1 - (abs(distance) - car_length)/(safe_distance - car_length))
        #a = 100/car_argmax[0]
        a = np.sign(car_argmax[0])*car_cost*a_max
    if lane_argmax is not None:
        if lane_argmax[1] != 0:
            b = 0.001/lane_argmax[1]

    return a, b


def get_mad(index, force_cst, verbose=False):
    total_distance = 0
    total_nb_collisions = 0
    total_arrived_to_dst = 0

    splits = torch.load('../splits.pth')
    n_indexes = len(splits[index])
    #n_indexes = 200
    dataloader = DataLoader(None, opt, 'i80')

    for j in range(n_indexes):
    #for j in range(20):
        car_path = dataloader.ids[splits[index][j]]
        timeslot, car_id = utils.parse_car_path(car_path)
        if verbose:
            print("Starting episode {}/{} with timeslot {}, car_id {}".format(j, n_indexes, timeslot, car_id))
        observation = env.reset(time_slot=timeslot, vehicle_id=car_id)

        done = False
        a, b = 0., 0.

        first_frame = True
        
        car_length = 12.
        while True: 
            input_images, input_states = observation['context'].contiguous(), observation['state'].contiguous()
            speed = input_states[-1:][:, 2:].norm(2, 1)

            a, b = action_SGD(input_images[-1:], input_states[-1:], force_cst, car_length)
            a = np.amax([a, -speed.numpy()/opt.delta_t])

            observation, reward, done, info =   env.step(np.array((a,b)))
            if first_frame:
                car_length = info._length
                first_frame_pos = info._position[0]
                first_frame = False
            if done or reward['collisions_per_frame'] > 0:
                break


            env.render()
        total_distance += info._position[0] - first_frame_pos 
        total_nb_collisions += info.collisions_per_frame
        total_arrived_to_dst += reward['arrived_to_dst']
        mad = PIXEL_METER_RATIO*float(total_distance)/(total_nb_collisions + 0.00001)
        print("MAD : {}".format(mad))
        print("arrived : {}/{}".format(total_arrived_to_dst, j+1))
        if verbose:
            print('distance travelled', info._position[0] - first_frame_pos)
            print('colisions', info.collisions_per_frame)

            print('Episode completed!')

    if verbose:
        print("Total MAD : {:2f} m".format(mad))
        print('Done')
    return mad, total_arrived_to_dst, n_indexes

#for i in [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]:
#for i in np.arange(0.005, 0.01, 0.001):
for i in [3]:
    mad, total_arrived_to_dst, n_indexes = get_mad('test_indx', i, verbose=False)
    print("mad for cst = {}, mad = {}".format(i, mad))
    with open("test_results.txt", "a") as results:
        results.write("cst : {}, mad : {}, arrived_to_dst : {}/{} \n".format(i, mad, total_arrived_to_dst, n_indexes))
