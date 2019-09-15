import argparse, os, re
import random
import torch
import numpy
import gym
from os import path
import planning
import utils
from dataloader import DataLoader
from imageio import imwrite


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-map', type=str, default='i80', help=' ')
parser.add_argument('-v', type=str, default='0', help=' ')
parser.add_argument('-seed', type=int, default=333333, help=' ')
# planning params
parser.add_argument('-method', type=str, default='policy-MPUR', help='[bprop|policy-MPUR|policy-MPER|policy-IL]')
parser.add_argument('-batch_size', type=int, default=1, help=' ')
parser.add_argument('-n_batches', type=int, default=200, help=' ')
parser.add_argument('-lrt', type=float, default=0.01, help=' ')
parser.add_argument('-ncond', type=int, default=20, help=' ')
parser.add_argument('-npred', type=int, default=30, help=' ')
parser.add_argument('-nexec', type=int, default=1, help=' ')
parser.add_argument('-n_rollouts', type=int, default=10, help=' ')
parser.add_argument('-rollout_length', type=int, default=1, help=' ')
parser.add_argument('-bprop_niter', type=int, default=5, help=' ')
parser.add_argument('-bprop_lrt', type=float, default=0.1, help=' ')
parser.add_argument('-bprop_buffer', type=int, default=1, help=' ')
parser.add_argument('-bprop_save_opt_stats', type=int, default=1, help=' ')
parser.add_argument('-n_dropout_models', type=int, default=10, help=' ')
parser.add_argument('-opt_z', type=int, default=0, help=' ')
parser.add_argument('-opt_a', type=int, default=1, help=' ')
parser.add_argument('-u_reg', type=float, default=0.0, help=' ')
parser.add_argument('-u_hinge', type=float, default=1.0, help=' ')
parser.add_argument('-lambda_l', type=float, default=0.0, help=' ')
# TODO: Add lambda_tl to parser.add_argument
parser.add_argument('-graph_density', type=float, default=0.001, help=' ')
parser.add_argument('-display', type=int, default=0, help=' ')
parser.add_argument('-debug', action='store_true', help=' ')
parser.add_argument('-model_dir', type=str, default='models/', help=' ')
M1 = 'model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-' + \
     'beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model'
M2 = 'model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-' + \
     'beta=1e-06-zdropout=0.0-gclip=5.0-warmstart=1-seed=1.step200000.model'
M3 = 'model=fwd-cnn-ten3-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-nhidden=128-fgeom=1-' + \
     'zeroact=0-zmult=0-dropout=0.1-nz=32-beta=0.0-zdropout=0.0-gclip=5.0-warmstart=1-seed=1.step200000.model'
M4 = 'model=fwd-cnn-ten3-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-nhidden=128-fgeom=1-' + \
     'zeroact=0-zmult=0-dropout=0.1-nz=32-beta=0.0-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model'
parser.add_argument('-mfile', type=str, default=M1, help=' ')
parser.add_argument('-value_model', type=str, default='', help=' ')
parser.add_argument('-policy_model', type=str, default='', help=' ')
parser.add_argument('-save_sim_video', action='store_true', help='Save simulator video in <frames> info attribute')

opt = parser.parse_args()

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)

opt.save_dir = path.join(opt.model_dir, 'planning_results')
opt.height = 117
opt.width = 24
opt.h_height = 14
opt.h_width = 3
opt.opt_z = (opt.opt_z == 1)
opt.opt_a = (opt.opt_a == 1)

data_path = f'traffic-data/state-action-cost/data_{opt.map}_v{opt.v}'


def load_models():
    stats = torch.load(path.join(data_path, 'data_stats.pth'))
    forward_model = torch.load(path.join(opt.model_dir, opt.mfile))
    if type(forward_model) is dict: forward_model = forward_model['model']
    value_function, policy_network_il, policy_network_mper = None, None, None
    model_path = path.join(opt.model_dir, f'policy_networks/{opt.policy_model}')
    if opt.value_model != '':
        value_function = torch.load(path.join(opt.model_dir, f'value_functions/{opt.value_model}')).cuda()
        forward_model.value_function = value_function
    if opt.method == 'policy-IL':
        policy_network_il = torch.load(model_path).cuda()
        policy_network_il.stats = stats
    if opt.method == 'policy-MPER':
        policy_network_mper = torch.load(model_path)['model']
        policy_network_mper.stats = stats
        forward_model.policy_net = policy_network_mper.policy_net
        forward_model.policy_net.stats = stats
        forward_model.policy_net.actor_critic = False
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


dataloader = DataLoader(None, opt, opt.map)
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

env = gym.make(env_names[opt.map])

plan_file = ''

if 'bprop' in opt.method:
    plan_file += opt.method
    if 'vae3' in opt.mfile:
        plan_file += f'-model=vae'
    elif 'ten3' in opt.mfile:
        plan_file += f'-model=ten'
    if 'zdropout=0.5' in opt.mfile:
        plan_file += '-zdropout=0.5'
    elif 'zdropout=0.0' in opt.mfile:
        plan_file += '-zdropout=0.0'
    if 'inferz=0' in opt.mfile:
        plan_file += '-inferz=0'
    elif 'inferz=1' in opt.mfile:
        plan_file += '-inferz=1'
    if 'deterministic' in opt.policy_model:
        plan_file += '-deterministic'
    if 'learnedcost=1' in opt.policy_model:
        plan_file += '-learnedcost=1'
    elif 'learnedcost=0' in opt.policy_model:
        plan_file += '-learnedcost=0'

    plan_file += f'-rollouts={opt.n_rollouts}'
    plan_file += f'-rollout_length={opt.npred}'
    plan_file += f'-lrt={opt.bprop_lrt}'
    plan_file += f'-niter={opt.bprop_niter}'
    plan_file += f'-ureg={opt.u_reg}'
    plan_file += f'-uhinge={opt.u_hinge}'
    plan_file += f'-n_dropout={opt.n_dropout_models}'
    plan_file += f'-abuffer={opt.bprop_buffer}'
    plan_file += f'-saveoptstats={opt.bprop_save_opt_stats}'
    plan_file += f'-lambdal={opt.lambda_l}'
    if opt.value_model != '':
        plan_file += f'-vmodel'
    plan_file += '-'

plan_file += f'{opt.policy_model}'

print(f'[saving to {path.join(opt.save_dir, plan_file)}]')

# different performance metrics
time_travelled, distance_travelled, road_completed, action_sequences, state_sequences = [], [], [], [], []

# n_test = len(splits['test_indx'])
# 1, 36, 15, 21, 5
car_ids = [
    "traffic-data/state-action-cost/data_i80_v0/trajectories-0400-0415/car21.pkl",
    "traffic-data/state-action-cost/data_i80_v0/trajectories-0400-0415/car5.pkl",
]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# for j in range(n_test):
for c in car_ids:
    # movie_dir = path.join(opt.save_dir, 'videos_simulator', plan_file, f'ep{j + 1}')
    movie_dir = path.join(opt.save_dir, 'videos_simulator', plan_file, f'ep{c.split("/")[-1]}')
    print(f'[new episode, will save to: {movie_dir}]')
    # car_path = dataloader.ids[splits['test_indx'][j]]
    car_path = c
    # TODO: Check timeslot and how it comes in (i.e. int == 0?)
    timeslot, car_id = utils.parse_car_path(car_path)
    # TODO: load car_id.pkl file to get the current lane dump so we can set target lane as a few seconds in the future
    inputs = env.reset(time_slot=timeslot, vehicle_id=car_id)  # if None => picked at random
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
#        input_images, input_states = inputs[0].contiguous(), inputs[1].contiguous()
        if opt.method == 'no-action':
            a = numpy.zeros((1, 2))
        elif opt.method == 'bprop':
            # TODO: car size is provided by the dataloader!! This lines below should be removed!
            # TODO: Namely, dataloader.car_sizes[timeslot][car_id]
            car_size = torch.Tensor([info._width / (0.3048 * 24 / 3.7),
                                     info._length / (0.3048 * 24 / 3.7)]).view(1, 2).cuda()
            a = planning.plan_actions_backprop(
                forward_model, input_images, input_states, car_size, npred=opt.npred, n_futures=opt.n_rollouts,
                normalize=True, bprop_niter=opt.bprop_niter, bprop_lrt=opt.bprop_lrt, u_reg=opt.u_reg,
                use_action_buffer=(opt.bprop_buffer == 1), n_models=opt.n_dropout_models,
                save_opt_stats=(opt.bprop_save_opt_stats == 1), nexec=opt.nexec, lambda_l=opt.lambda_l
            )
        elif opt.method == 'policy-IL':
            _, _, _, a = policy_network_il(input_images, input_states, sample=True,
                                           normalize_inputs=True, normalize_outputs=True)
            a = a.squeeze().cpu().view(1, 2).numpy()
        elif opt.method == 'policy-MPER':
            a, entropy, mu, std = forward_model.policy_net(input_images, input_states, sample=True,
                                                           normalize_inputs=True, normalize_outputs=True)
            a = a.cpu().view(1, 2).numpy()
        elif opt.method == 'policy-MPUR':
            target_y = torch.tensor(48, dtype=torch.float).to(device)  # middle of lane 4 (0 index) == pixel 144
            a, entropy, mu, std = forward_model.policy_net(input_images, input_states, sample=True,
                                                           normalize_inputs=True, normalize_outputs=True,
                                                           controls=dict(target_lanes=target_y))
            a = a.cpu().view(1, 2).numpy()
        elif opt.method == 'bprop+policy-IL':
            _, _, _, a = policy_network_il(input_images, input_states, sample=True,
                                           normalize_inputs=True, normalize_outputs=False)
            a = a[0]
            a = forward_model.plan_actions_backprop(input_images, input_states, npred=opt.npred,
                                                    n_futures=opt.n_rollouts, normalize=True,
                                                    bprop_niter=opt.bprop_niter, bprop_lrt=opt.bprop_lrt,
                                                    actions=a, u_reg=opt.u_reg, nexec=opt.nexec)

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
            print('(action: ({:.4f}, {:.4f}) | true costs: (prox: {:.4f}, lane: {:.4f}, target_lane: {:.4f})]'.format(
                a[t][0], a[t][1], cost['pixel_proximity_cost'], cost['lane_cost'], cost['target_lane_cost'])
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
        # f'ep: {j + 1:3d}/{n_test}',
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
            os.mkdir(sim_path)
            for n, img in enumerate(info.frames):
                imwrite(path.join(sim_path, f'im{n:05d}.png'), img)
