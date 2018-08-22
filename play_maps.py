import argparse
import numpy
import gym

parser = argparse.ArgumentParser()
parser.add_argument('-nb_conditions', type=int, default=10)
parser.add_argument('-display', type=int, default=1)
parser.add_argument('-map', type=str, default='i80', choices={'ai', 'i80', 'us101', 'lanker', 'peach'})
parser.add_argument('-state_image', type=int, default=0)
parser.add_argument('-store', type=int, default=0)
parser.add_argument('-nb_episodes', type=int, default=1)
parser.add_argument('-fps', type=int, default=1e3)
parser.add_argument('-delta_t', type=float, default=0.1)

opt = parser.parse_args()

kwargs = {
    'fps': opt.fps,
    'nb_states': opt.nb_conditions,
    'display': opt.display,
    'state_image': opt.state_image,
    'store': opt.store,
    'delta_t': opt.delta_t,
}

gym.envs.registration.register(
    id='Traffic-v0',
    entry_point='traffic_gym:Simulator',
    kwargs=kwargs
)

gym.envs.registration.register(
    id='I-80-v0',
    entry_point='map_i80:I80',
    kwargs=kwargs,
)

gym.envs.registration.register(
    id='US-101-v0',
    entry_point='map_us101:US101',
    kwargs=kwargs,
)

gym.envs.registration.register(
    id='Lankershim-v0',
    entry_point='map_lanker:Lankershim',
    kwargs=kwargs,
)

gym.envs.registration.register(
    id='Peachtree-v0',
    entry_point='map_peach:Peachtree',
    kwargs=kwargs,
)

env_names = {
    'ai': 'Traffic-v0',
    'i80': 'I-80-v0',
    'us101': 'US-101-v0',
    'lanker': 'Lankershim-v0',
    'peach': 'Peachtree-v0',
}

print('Building the environment (loading data, if any)')
env = gym.make(env_names[opt.map])

for episode in range(opt.nb_episodes):
    # env.reset(frame=int(input('Frame: ')), time_slot=0)
    env.reset(frame=0, time_slot=0)

    done = False
    while not done:
        observation, reward, done, info = env.step(numpy.zeros((2,)))
        # print(observation, reward, done, info)
        env.render()

    print('Episode completed!')

print('Done')
