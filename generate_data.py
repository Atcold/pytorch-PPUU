import torch, random, math, argparse, pickle, os, gym
import numpy as np
import traffic_gym  # initialise the gym environment
from gym.envs.registration import register

parser = argparse.ArgumentParser()
parser.add_argument('-display', type=int, default=0)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-lanes', type=int, default=3)
parser.add_argument('-n_episodes', type=int, default=10000)
parser.add_argument('-data_dir', type=str, default='data/')
opt = parser.parse_args()

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

os.system("mkdir -p " + opt.data_dir)

data_file = f'{opt.data_dir}/traffic_data_lanes={opt.lanes}-episodes={opt.n_episodes}-seed={opt.seed}.pkl'
print(f'Will save as {data_file}')

register(
    id='Traffic-v0',
    entry_point='traffic_gym:StatefulEnv',
    tags={'wrapper_config.TimeLimit.max_episodesteps': 100},
    kwargs={'display': opt.display,
            'nb_lanes': opt.lanes},
)

env = gym.make('Traffic-v0')
env.reset()


# class PID:
#     def __init__(self, target):
#         self.Kp = 0.003 + np.random.normal() * 0.001
#         self.Kd = 0.002 + np.random.normal() * 0.001
#         self.last_err_x = 0
#         self.last_err_y = 0
#         self.target = target
#
#     @staticmethod
#     def _detect_collision(object1, object2):
#         dx = object1.x - object2.x
#         dy = object1.y - object2.y
#         distance = math.hypot(dx, dy)
#         if distance < (object1.r + object2.r):
#             return True
#         return False
#
#     def act(self, ship, planets, waypoints):
#         # compute errors and error derivatives
#         err_x = waypoints[self.target].x - ship.x
#         err_y = waypoints[self.target].y - ship.y
#         derr_x = err_x - self.last_err_x
#         derr_y = err_y - self.last_err_y
#         self.last_err_x = err_x
#         self.last_err_y = err_y
#
#         # compute continuous control (PD controller)
#         ux = self.Kp * err_x + self.Kd * derr_x
#         uy = self.Kp * err_y + self.Kd * derr_y
#         norm = math.sqrt(ux ** 2 + uy ** 2)
#         u_clip = 0.15
#         if norm > u_clip:
#             ux /= (norm / u_clip)
#             uy /= (norm / u_clip)
#
#         discrete_action = 0  # no-op by default
#         # if it reaches a waypoint, executes random action
#         reached_waypoint = False
#         for k in range(len(waypoints)):
#             if self._detect_collision(waypoints[k], ship):
#                 reached_waypoint = True
#                 if random.random() > 0.5:
#                     if random.random() > 0.5:
#                         discrete_action = k + 1
#                     else:
#                         discrete_action = random.choice([x for x in [0, 1, 2, 3] if x != k + 1])
#
#         # occasionally drops cargo
#         if not reached_waypoint:
#             if random.random() > 0.999:
#                 discrete_action = random.randint(1, 3)
#
#         one_hot = np.zeros(4)
#         one_hot[discrete_action] = 1
#         a = np.concatenate((np.array((ux, uy)), one_hot))
#         return a


def run_episode():
    # target = random.randint(0, 3 - 1)
    # agent = PID(target)
    action = np.array([0, 0, 1, 0, 0, 0])
    states, actions, rewards = [], [], []
    # exit_condition = 0
    done = False

    state, objects = env.reset()
    states.append(state)
    while not done:
        # ship, planets, waypoints = objects
        # action = agent.act(ship, planets, waypoints)
        state, reward, done, objects = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        env.render()

    return states, actions, rewards


episodes = []
for i in range(opt.n_episodes):
    print(f'episode {i + 1}/{opt.n_episodes}')
    states, actions, rewards = run_episode()
    episodes.append({'states': states, 'actions': actions, 'rewards': rewards})

pickle.dump(episodes, open(data_file, 'wb'))
