import argparse
import pdb
import numpy
import gym

parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-nb_conditions', type=int, default=10)
parser.add_argument('-nb_samples', type=int, default=1)
parser.add_argument('-display', type=int, default=1)
parser.add_argument('-map', type=str, default='i80', choices={'i80', 'us101', 'lanker', 'peach'})
parser.add_argument('-fps', type=int, default=1e3)
parser.add_argument('-delta_t', type=float, default=0.1)
parser.add_argument('-nb_episodes', type=int, default=10)

opt = parser.parse_args()

kwargs = {
    'fps': opt.fps,
    'nb_states': opt.nb_conditions,
    'display': opt.display,
    'delta_t': opt.delta_t,
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

for episode in range(opt.nb_episodes):
    context, states = list(), list()

    observation = env.reset(time_slot=None, vehicle_id=None)  # if None => picked at random
    context.append(observation['context'])
    states.append(observation['state'])

    done = False
    while not done:
        observation, reward, done, info = env.step(numpy.zeros((2,)))

        context.append(observation['context'])
        states.append(observation['state'])
        # print(observation, reward, done, info)

        env.render()

    print('Episode completed!')
    pdb.set_trace()

print('Done')
