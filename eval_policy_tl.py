import copy
import random
from os import mkdir, path

import gym
import numpy
import torch
from imageio import imwrite

import planning
import utils
from dataloader import DataLoader

opt = utils.parse_command_line()
random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)
device = torch.device('cuda' if torch.cuda.is_available() and not opt.no_cuda else 'cpu')

opt.save_dir = path.join(opt.model_dir, 'planning_results')
opt.height = 117
opt.width = 24
opt.h_height = 14
opt.h_width = 3

data_path = f'traffic-data/state-action-cost/data_{opt.dataset}_v{opt.v}'


def load_models():
    stats = torch.load(path.join(data_path, 'data_stats.pth'))
    forward_model = torch.load(path.join(opt.model_dir, opt.mfile))
    if type(forward_model) is dict: forward_model = forward_model['model']
    value_function, policy_network_il, policy_network_mper = None, None, None
    model_path = path.join(opt.model_dir, f'policy_networks/{opt.policy_model}')
    if opt.method == 'policy-MPUR':
        policy_network_mpur = torch.load(model_path)['model']
        policy_network_mpur.stats = stats
        forward_model.policy_net = policy_network_mpur.policy_net
        forward_model.policy_net.stats = stats
        forward_model.policy_net.actor_critic = False

    forward_model.intype('gpu')
    forward_model.stats = stats
    if 'ten' in opt.mfile:
        forward_model.p_z = torch.load(path.join(opt.model_dir, f'{opt.mfile}.pz'))
    return forward_model, value_function, policy_network_il, policy_network_mper, stats


dataloader = DataLoader(None, opt, opt.dataset)
forward_model, value_function, policy_network_il, policy_network_mper, data_stats = load_models()
splits = torch.load(path.join(data_path, 'splits.pth'))

if opt.u_reg > 0.0:
    forward_model.train()
    forward_model.opt.u_hinge = opt.u_hinge
    if hasattr(forward_model, 'value_function'):
        forward_model.value_function.train()
    planning.estimate_uncertainty_stats(forward_model, dataloader, n_batches=50, npred=opt.npred)

gym.envs.registration.register(
    id='I-80-v1',
    entry_point='map_i80_ctrl:ControlledI80',
    kwargs=dict(
        fps=10,
        nb_states=opt.ncond,
        display=False,
        delta_t=0.1,
        store_simulator_video=opt.save_sim_video,
    )
)

print('Building the environment (loading data, if any)')
env_names = {
    'i80': 'I-80-v1',
}

env = gym.make(env_names[opt.dataset])
plan_file = f'{opt.policy_model}'

print(f'[saving to {path.join(opt.save_dir, plan_file)}]')

# different performance metrics
time_travelled, distance_travelled, road_completed, action_sequences, state_sequences = [], [], [], [], []

n_test = len(splits['test_indx'])
for j in range(n_test):
    movie_dir = path.join(opt.save_dir, 'videos_simulator', plan_file, f'ep{j + 1}')
    print(f'[new episode, will save to: {movie_dir}]')
    car_path = dataloader.ids[splits['test_indx'][j]]
    timeslot, car_id = utils.parse_car_path(car_path)
    inputs, info = env.reset(time_slot=timeslot, vehicle_id=car_id)  # if None => picked at random
    car_trajectory = copy.deepcopy(info._trajectory)  # copy the test car's true trajectory
    forward_model.reset_action_buffer(opt.npred)
    done, mu, std = False, None, None
    images, states, costs, actions, mu_list, std_list = [], [], [], [], [], []
    cntr = 0
    # inputs, cost, done, info = env.step(numpy.zeros((2,)))
    input_state_t0 = inputs['state'].contiguous()[-1]
    action_sequences.append([])
    state_sequences.append([])
    while not done:
        input_images = inputs['context'].contiguous()
        input_states = inputs['state'].contiguous()
        if opt.method == 'policy-MPUR':
            # Target y = average of car's true y position from now and 30 (opt.npred) steps into the future
            target_y = torch.tensor(car_trajectory[cntr:cntr + opt.npred, 1].mean()).to(device)
            print(f'(cntr={cntr}) target y: {target_y}')
            a, entropy, mu, std = forward_model.policy_net(input_images, input_states, sample=True,
                                                           normalize_inputs=True, normalize_outputs=True,
                                                           controls=dict(target_lanes=target_y))
            a = a.cpu().view(1, 2).numpy()

        action_sequences[-1].append(a)
        state_sequences[-1].append(input_states)
        cntr += 1
        cost_test = 0
        t = 0
        T = opt.npred if opt.nexec == -1 else opt.nexec
        while (t < T) and not done:
            inputs, cost, done, info = env.step(a[t])
            if info.collisions_per_frame > 0:
                print(f'[collision after {cntr} frames, ending]')
                done = True
            print('(action: ({:.4f}, {:.4f}) | true costs: (prox: {:.4f}, lane: {:.4f})]'.format(
                a[t][0], a[t][1], cost['pixel_proximity_cost'], cost['lane_cost'])
            )

            images.append(input_images[-1])
            states.append(input_states[-1])
            costs.append([cost['pixel_proximity_cost'], cost['lane_cost']])
            if opt.mfile == 'no-action':
                actions.append(a[t])
                mu_list.append(mu)
                std_list.append(std)
            else:
                actions.append(((torch.tensor(a[t]) - data_stats['a_mean']) / data_stats['a_std']))
                if mu is not None:
                    mu_list.append(mu.data.cpu().numpy())
                    std_list.append(std.data.cpu().numpy())
            t += 1
        costs_ = numpy.stack(costs)
    input_state_tfinal = inputs['state'][-1]
    time_travelled.append(len(images))
    distance_travelled.append(input_state_tfinal[0] - input_state_t0[0])
    road_completed.append(1 if cost['arrived_to_dst'] else 0)
    log_string = ' | '.join((
        f'ep: {j + 1:3d}/{n_test}',
        f'time: {time_travelled[-1]}',
        f'distance: {distance_travelled[-1]:.0f}',
        f'success: {road_completed[-1]:d}',
        f'mean time: {torch.Tensor(time_travelled).mean():.0f}',
        f'mean distance: {torch.Tensor(distance_travelled).mean():.0f}',
        f'mean success: {torch.Tensor(road_completed).mean():.3f}',
    ))
    print(log_string)
    utils.log(path.join(opt.save_dir, f'{plan_file}.log'), log_string)
    torch.save(action_sequences, path.join(opt.save_dir, f'{plan_file}.actions'))
    torch.save(state_sequences, path.join(opt.save_dir, f'{plan_file}.states'))

    images  = torch.stack(images)
    states  = torch.stack(states)
    costs   = torch.tensor(costs)
    actions = torch.stack(actions)

    if mu is not None:
        mu_list = numpy.stack(mu_list)
        std_list = numpy.stack(std_list)
    else:
        mu_list, std_list = None, None

    if len(images) > 3:
        utils.save_movie(path.join(movie_dir, 'ego'), images.float() / 255.0, states, costs,
                         actions=actions, mu=mu_list, std=std_list, pytorch=True)
        if opt.save_sim_video:
            sim_path = path.join(movie_dir, 'sim')
            print(f'[saving simulator movie to {sim_path}]')
            mkdir(sim_path)
            for n, img in enumerate(info.frames):
                imwrite(path.join(sim_path, f'im{n:05d}.png'), img)
