import os

# These environment variables need to be set before
# import numpy to prevent numpy from spawning a lot of processes
# which end up clogging up the system.
os.environ["NUMEXPR_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"

import argparse
import random
import torch
import torch.nn
import torch.nn.parallel
import numpy
import gym
from os import path
import planning
import utils
from dataloader import DataLoader
from imageio import imwrite
import time

from torch.multiprocessing import Pool, set_start_method

torch.multiprocessing.set_sharing_strategy('file_system')


class SimulationResult:

    def __init__(self):
        self.action_sequence = None
        self.state_sequence = None
        self.road_completed = None
        self.distance_travelled = None
        self.time_travelled = None
        self.has_collided = None
        self.off_screen = None

    @staticmethod
    def dummy():
        result = SimulationResult()
        result.action_sequence = torch.rand((20, 800))
        result.state_sequence = []
        result.road_completed = 1
        result.distance_travelled = 100
        result.time_travelled = 100
        result.has_collided = 0
        result.off_screen = 0
        return result


def get_optimal_pool_size():
    available_processes = len(os.sched_getaffinity(0))
    # we can't use more than 10, as in that case we don't fit into Gpu.
    optimal_pool_size = min(10, available_processes)
    return optimal_pool_size


def load_models(opt, data_path, device='cuda'):
    stats = torch.load(path.join(data_path, 'data_stats.pth'))

    model_path = path.join(opt.model_dir, opt.mfile)
    if path.exists(model_path):
        forward_model = torch.load(model_path)
    elif path.exists(opt.mfile):
        forward_model = torch.load(opt.mfile)
    else:
        raise runtime_error(f'couldn\'t find file {opt.mfile}')

    if type(forward_model) is dict:
        forward_model = forward_model['model']
    value_function, policy_network_il, policy_network_mper = None, None, None
    model_path = path.join(
        opt.model_dir,
        f'policy_networks/{opt.policy_model}'
    )
    if opt.value_model != '':
        value_function = torch.load(
            path.join(
                opt.model_dir,
                f'value_functions/{opt.value_model}')
        ).cuda()
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
        checkpoint = torch.load(model_path)
        policy_network_mpur = checkpoint['model']
        policy_network_mpur.stats = stats
        forward_model.policy_net = policy_network_mpur.policy_net
        forward_model.policy_net.stats = stats
        forward_model.policy_net.actor_critic = False
        forward_model.policy_net.options = checkpoint['opt']

    forward_model.intype('gpu')
    forward_model.stats = stats
    if hasattr(forward_model, 'policy_net'):
        forward_model.policy_net.stats_d = {}
        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                forward_model.policy_net.stats_d[k] = v.to(device)

    forward_model = forward_model.share_memory()

    if 'ten' in opt.mfile:
        forward_model.p_z = torch.load(
            path.join(opt.model_dir, f'{opt.mfile}.pz'))
    return (
        forward_model,
        value_function,
        policy_network_il,
        policy_network_mper,
        stats
    )


def build_plan_file_name(opt):
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
        plan_file += f'-lambdao={opt.lambda_o}'
        if opt.value_model != '':
            plan_file += f'-vmodel'
        plan_file += '-'

    plan_file += f'{opt.policy_model}'
    return plan_file


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@'
    )
    parser.add_argument('-map', type=str, default='i80', help=' ')
    parser.add_argument('-v', type=str, default='3', help=' ')
    parser.add_argument('-seed', type=int, default=333333, help=' ')
    # planning params
    parser.add_argument('-method', type=str, default='bprop',
                        help='[bprop|policy-MPUR|policy-MPER|policy-IL]')
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
    parser.add_argument('-lambda_o', type=float, default=0.0, help=' ')
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
    M5 = 'model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-' + \
         'beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step400000.model'
    parser.add_argument('-mfile', type=str, default=M5, help=' ')
    parser.add_argument('-value_model', type=str, default='', help=' ')
    parser.add_argument('-policy_model', type=str, default='', help=' ')
    parser.add_argument('-save_sim_video', action='store_true',
                        help='Save simulator video in <frames> info attribute')
    parser.add_argument('-enable_tensorboard', action='store_true',
                        help='Enables tensorboard logging.')
    parser.add_argument('-tensorboard_dir',
                        type=str,
                        default='models/planning_results',
                        help='path to the directory where to save tensorboard log.' +
                             'If passed empty path no logs are saved.')
    parser.add_argument('-num_processes', type=int, default=-1,
                        help='number of workers to use. Default is min(10, #cores).')
    parser.add_argument('-save_grad_vid',
                        action='store_true',
                        help='save gradients wrt states')

    opt = parser.parse_args()
    opt.save_dir = path.join(opt.model_dir, 'planning_results')
    opt.height = 117
    opt.width = 24
    opt.h_height = 14
    opt.h_width = 3
    opt.opt_z = (opt.opt_z == 1)
    opt.opt_a = (opt.opt_a == 1)

    if opt.num_processes == -1:
        opt.num_processes = get_optimal_pool_size()

    return opt


def process_one_episode(opt,
                        env,
                        car_path,
                        forward_model,
                        policy_network_il,
                        data_stats,
                        plan_file,
                        index,
                        car_sizes):
    movie_dir = path.join(
        opt.save_dir, 'videos_simulator', plan_file, f'ep{index + 1}')
    if opt.save_grad_vid:
        grad_movie_dir = path.join(
            opt.save_dir, 'grad_videos_simulator', plan_file, f'ep{index + 1}')
        print(f'[gradient videos will be saved to: {grad_movie_dir}]')
    timeslot, car_id = utils.parse_car_path(car_path)
    # if None => picked at random
    inputs = env.reset(time_slot=timeslot, vehicle_id=car_id)
    forward_model.reset_action_buffer(opt.npred)
    done, mu, std = False, None, None
    images, states, costs, actions, mu_list, std_list, grad_list = [], [], [], [], [], [], []
    cntr = 0
    # inputs, cost, done, info = env.step(numpy.zeros((2,)))
    input_state_t0 = inputs['state'].contiguous()[-1]
    cost_sequence, action_sequence, state_sequence = [], [], []
    has_collided = False
    off_screen = False
    while not done:
        input_images = inputs['context'].contiguous()
        input_states = inputs['state'].contiguous()
        if opt.save_grad_vid:
            grad_list.append(planning.get_grad_vid(
                forward_model, input_images, input_states,
                car_sizes,
                device='cuda' if torch.cuda.is_available else 'cpu'
            ))
        if opt.method == 'no-action':
            a = numpy.zeros((1, 2))
        elif opt.method == 'bprop':
            # TODO: car size is provided by the dataloader!! This lines below should be removed!
            # TODO: Namely, dataloader.car_sizes[timeslot][car_id]
            a = planning.plan_actions_backprop(
                forward_model,
                input_images[:, :3, :, :].contiguous(),
                input_states,
                car_sizes,
                npred=opt.npred,
                n_futures=opt.n_rollouts,
                normalize=True,
                bprop_niter=opt.bprop_niter,
                bprop_lrt=opt.bprop_lrt,
                u_reg=opt.u_reg,
                use_action_buffer=(opt.bprop_buffer == 1),
                n_models=opt.n_dropout_models,
                save_opt_stats=(opt.bprop_save_opt_stats == 1),
                nexec=opt.nexec,
                lambda_l=opt.lambda_l,
                lambda_o=opt.lambda_o
            )
        elif opt.method == 'policy-IL':
            _, _, _, a = policy_network_il(
                input_images,
                input_states,
                sample=True,
                normalize_inputs=True,
                normalize_outputs=True
            )
            a = a.squeeze().cpu().view(1, 2).numpy()
        elif opt.method == 'policy-MPER':
            a, entropy, mu, std = forward_model.policy_net(
                input_images,
                input_states,
                sample=True,
                normalize_inputs=True,
                normalize_outputs=True
            )
            a = a.cpu().view(1, 2).numpy()
        elif opt.method == 'policy-MPUR':
            a, entropy, mu, std = forward_model.policy_net(
                input_images,
                input_states,
                sample=True,
                normalize_inputs=True,
                normalize_outputs=True
            )
            a = a.cpu().view(1, 2).numpy()
        elif opt.method == 'bprop+policy-IL':
            _, _, _, a = policy_network_il(
                input_images,
                input_states,
                sample=True,
                normalize_inputs=True,
                normalize_outputs=False
            )
            a = a[0]
            a = forward_model.plan_actions_backprop(
                input_images,
                input_states,
                npred=opt.npred,
                n_futures=opt.n_rollouts,
                normalize=True,
                bprop_niter=opt.bprop_niter,
                bprop_lrt=opt.bprop_lrt,
                actions=a,
                u_reg=opt.u_reg,
                nexec=opt.nexec
            )

        action_sequence.append(a)
        state_sequence.append(input_states)
        cntr += 1
        cost_test = 0
        t = 0
        T = opt.npred if opt.nexec == -1 else opt.nexec
        while (t < T) and not done:
            inputs, cost, done, info = env.step(a[t])
            if info.collisions_per_frame > 0:
                has_collided = True
                # print(f'[collision after {cntr} frames, ending]')
                done = True
            off_screen = info.off_screen

            images.append(input_images[-1])
            states.append(input_states[-1])
            costs.append([cost['pixel_proximity_cost'], cost['lane_cost']])
            cost_sequence.append(cost)
            if opt.mfile == 'no-action':
                actions.append(a[t])
                mu_list.append(mu)
                std_list.append(std)
            else:
                actions.append(
                    ((torch.tensor(a[t]) - data_stats['a_mean'])
                        / data_stats['a_std'])
                )
                if mu is not None:
                    mu_list.append(mu.data.cpu().numpy())
                    std_list.append(std.data.cpu().numpy())
            t += 1
    input_state_tfinal = inputs['state'][-1]

    if mu is not None:
        mu_list = numpy.stack(mu_list)
        std_list = numpy.stack(std_list)
    else:
        mu_list, std_list = None, None

    images = torch.stack(images)
    states = torch.stack(states)
    costs = torch.tensor(costs)
    actions = torch.stack(actions)
    if opt.save_grad_vid:
        grads = torch.cat(grad_list)

    if len(images) > 3:
        images_3_channels = (images[:, :3] + images[:, 3:]).clamp(max=255)
        utils.save_movie(path.join(movie_dir, 'ego'),
                         images_3_channels.float() / 255.0,
                         states,
                         costs,
                         actions=actions,
                         mu=mu_list,
                         std=std_list,
                         pytorch=True)
        if opt.save_grad_vid:
            utils.save_movie(
                grad_movie_dir,
                grads,
                None,
                None,
                None,
                None,
                None,
                pytorch=True
            )
        if opt.save_sim_video:
            sim_path = path.join(movie_dir, 'sim')
            print(f'[saving simulator movie to {sim_path}]')
            os.mkdir(sim_path)
            for n, img in enumerate(info.frames):
                imwrite(path.join(sim_path, f'im{n:05d}.png'), img)

    returned = SimulationResult()
    returned.time_travelled = len(images)
    returned.distance_travelled = input_state_tfinal[0] - input_state_t0[0]
    returned.road_completed = 1 if cost['arrived_to_dst'] else 0
    returned.off_screen = off_screen
    returned.has_collided = has_collided
    returned.action_sequence = numpy.stack(action_sequence)
    returned.state_sequence = numpy.stack(state_sequence)
    returned.cost_sequence = numpy.stack(cost_sequence)

    return returned


def main():
    opt = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    data_path = 'traffic-data/state-action-cost/data_i80_v0'

    dataloader = DataLoader(None, opt, 'i80')
    (
        forward_model,
        value_function,
        policy_network_il,
        policy_network_mper,
        data_stats
    ) = load_models(opt, data_path, device)
    splits = torch.load(path.join(data_path, 'splits.pth'))

    if opt.u_reg > 0.0:
        forward_model.train()
        forward_model.opt.u_hinge = opt.u_hinge
        if hasattr(forward_model, 'value_function'):
            forward_model.value_function.train()
        planning.estimate_uncertainty_stats(
            forward_model, dataloader, n_batches=50, npred=opt.npred)

    gym.envs.registration.register(
        id='I-80-v1',
        entry_point='map_i80_ctrl:ControlledI80',
        kwargs=dict(
            fps=10,
            nb_states=opt.ncond,
            display=False,
            delta_t=0.1,
            store_simulator_video=opt.save_sim_video,
            show_frame_count=False,
        )
    )

    print('Building the environment (loading data, if any)')
    env_names = {
        'i80': 'I-80-v1',
    }
    env = gym.make(env_names[opt.map])

    plan_file = build_plan_file_name(opt)
    print(f'[saving to {path.join(opt.save_dir, plan_file)}]')

    # different performance metrics
    time_travelled, distance_travelled, road_completed = [], [], []
    # values saved for later inspection
    action_sequences, state_sequences, cost_sequences =  [], [], []

    writer = utils.create_tensorboard_writer(opt)

    n_test = len(splits['test_indx'])

    set_start_method('spawn')
    pool = Pool(opt.num_processes)

    async_results = []

    time_started = time.time()
    total_images = 0

    for j in range(n_test):
        car_path = dataloader.ids[splits['test_indx'][j]]
        timeslot, car_id = utils.parse_car_path(car_path)
        car_sizes = torch.tensor(
                    dataloader.car_sizes[sorted(list(dataloader.car_sizes.keys()))[
                        timeslot]][car_id]
                )[None, :]
        async_results.append(
            pool.apply_async(
                process_one_episode, (
                    opt,
                    env,
                    car_path,
                    forward_model,
                    policy_network_il,
                    data_stats,
                    plan_file,
                    j,
                    car_sizes
                )
            )
        )

    for j in range(n_test):
        simulation_result = async_results[j].get()

        time_travelled.append(simulation_result.time_travelled)
        distance_travelled.append(simulation_result.distance_travelled)
        road_completed.append(simulation_result.road_completed)
        action_sequences.append(torch.from_numpy(
            simulation_result.action_sequence))
        state_sequences.append(torch.from_numpy(
            simulation_result.state_sequence))
        cost_sequences.append(simulation_result.cost_sequence)
        total_images += time_travelled[-1]

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

        if writer is not None:
            # writer.add_video(
            #     f'Video/success={simulation_result.road_completed:d}_{j}',
            #     simulation_result.images.unsqueeze(0),
            #     j
            # )
            writer.add_scalar('ByEpisode/Success',
                              simulation_result.road_completed, j)
            writer.add_scalar('ByEpisode/Collision',
                              simulation_result.has_collided, j)
            writer.add_scalar('ByEpisode/OffScreen',
                              simulation_result.off_screen, j)
            writer.add_scalar('ByEpisode/Distance',
                              simulation_result.distance_travelled, j)

    pool.close()
    pool.join()

    diff_time = time.time() - time_started
    print('avg time travelled per second is', total_images / diff_time)

    torch.save(action_sequences, path.join(
        opt.save_dir, f'{plan_file}.actions'))
    torch.save(state_sequences, path.join(opt.save_dir, f'{plan_file}.states'))
    torch.save(cost_sequences, path.join(opt.save_dir, f'{plan_file}.costs'))

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
