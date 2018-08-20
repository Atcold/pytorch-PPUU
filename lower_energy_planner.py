import argparse
import numpy
import gym
import pdb

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

def distance(car1, car2):
    diff = car1.get_state()[:2] - car2.get_state()[:2]
    return torch.norm(diff)

def get_total_energy(env, autonomous_car):
    energy = 0
    for car in env.vehicles:
        if car.id is not autonomous_car.id:
            energy += 1./(distance(car, autonomous_car))
    return energy

for episode in range(1000):
    env.reset()

    done = False
    autonomous_car = env
    while not done:
        autonomous_car = get_controlled_car(env)
        if autonomous_car:
            print("total energy is %f" % get_total_energy(env, autonomous_car))
        observation, reward, done, info = env.step(numpy.array((0,0)))

        env.render()

    print('Episode completed!')

print('Done')
