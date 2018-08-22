import argparse
import numpy
import gym
import pdb
import torch

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
    energy_vector = numpy.array([1, 10])

    for car in env.vehicles:
        if car.id is not autonomous_car.id:
            cx, cy = car.get_state().numpy()[:2] + car.get_state().numpy()[2:]*car._dt
            ener = -10000*energy_vector[0]*(x - cx)/(energy_vector[0]*(cx - x)**2 + energy_vector[1]*(cy - y)**2)**2
            #if car.id == 700:
                #print(ener)
            tot += ener
    return tot

def get_total_energy(env, autonomous_car):
    energy = 0
    for car in env.vehicles:
        if car.id is not autonomous_car.id:
            energy += compute_energy(car.get_state()[:2], autonomous_car.get_state()[:2])
            print(energy)
    return energy

for episode in range(1000):
    env.reset()

    max_a = 30
    done = False
    a, b = 0., 0.
    while not done:
        autonomous_car = get_controlled_car(env)
        if autonomous_car:
            #print("total energy is %f" % get_total_energy(env, autonomous_car))
            x_o, y_o, dx_o, dy_o = autonomous_car.get_state().numpy()
            x, y = x_o, y_o
            #print("starting x = {}".format(x))
            #print(dx_o)
            for i in range(20):
                x -= 10*dE_dxj(env, autonomous_car.id, x, y)

            #print("x = {}".format(x))
            a = ((x - x_o)/ autonomous_car._dt -  dx_o) / autonomous_car._dt
            a = a if abs(a) < max_a else numpy.sign(a) * max_a

        observation, reward, done, info = env.step(numpy.array((a,0)))

        env.render()

    print('Episode completed!')

print('Done')
