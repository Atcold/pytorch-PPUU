import argparse
import os

import gym
import torch
from imageio import imwrite

import utils


def load_policy(device, data_path, model_dir, mfile, policy_model):
    """
    Load the policy model from disk
    :param device: cuda device or CPU
    :param data_path: path to traffic-data to get data stats
    :param model_dir: path to forward and policy model
    :param mfile: forward model
    :param policy_model: policy model
    :return: policy model and data stats
    """
    data_stats = torch.load(os.path.join(data_path, 'data_stats.pth'))
    model_path = os.path.join(model_dir, f'policy_networks/{policy_model}')

    forward_model = torch.load(os.path.join(model_dir, mfile))
    policy_network_mpur = torch.load(model_path, map_location=device)['model']
    policy_network_mpur.stats = data_stats
    forward_model.policy_net = policy_network_mpur.policy_net
    forward_model.policy_net.stats = data_stats
    forward_model.policy_net.actor_critic = False

    # policy_network_mpur = torch.load(model_path, map_location=device)['model']
    # policy_network_mpur.policy_net.stats = data_stats
    return forward_model, data_stats


def main():
    pm = 'MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-' \
                   'lambdal=0.2-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=False-learnedcost=True-lambdatl=1.0' \
                   '-seed=2-novaluestep90000.model'
    fm = 'model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-' + \
         'beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model'
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=3333)
    parser.add_argument('-v', type=str, default='0')
    parser.add_argument('-dataset', type=str, default='i80')
    parser.add_argument('-sim', type=str, default='simI80', choices={'simI80'})
    parser.add_argument('-model_dir', type=str, default='models_learned_cost')
    parser.add_argument('-forward_model', type=str, default=fm)
    parser.add_argument('-policy_model', type=str, default=pm)
    parser.add_argument('-ncond', type=int, default=20)
    parser.add_argument('-npred', type=int, default=20)
    parser.add_argument('-display', action='store_true')
    parser.add_argument('-store_screen', action='store_true')
    parser.add_argument('-save_sim_video', action='store_true')
    parser.add_argument('-fps', type=int, default=1e3)
    parser.add_argument('-delta_t', type=float, default=0.1)
    opt = parser.parse_args()

    # Build gym
    kwargs = {
        'fps': opt.fps,
        'nb_states': opt.ncond,
        'display': opt.display,
        'store_simulator_video': opt.save_sim_video,
        'store': opt.store_screen,
        'delta_t': opt.delta_t,
    }
    gym.envs.registration.register(
        id='I80-Simulation-v0',
        entry_point='map_i80_ctrl:SimI80',
        kwargs=kwargs,
    )
    env_names = {
        'simI80': 'I80-Simulation-v0',
    }
    print(f'Building the environment for {opt.sim}...')
    env = gym.make(env_names[opt.sim])
    movie_dir = os.path.join('simulator-dumps', 'train-of-cars', opt.policy_model)

    # Load policy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(opt.seed)  # TODO: Not sure if this is needed
    data_path = f'traffic-data/state-action-cost/data_{opt.dataset}_v{opt.v}'
    model, stats = load_policy(device, data_path, opt.model_dir, opt.forward_model, opt.policy_model)

    # Reset environment
    observation = env.reset(control=True)

    # Run simulation
    target_y = torch.tensor(96.0).to(device)
    images, states, actions = [], [], []
    action_sequences, state_sequences = [], []
    action_sequences.append([])
    state_sequences.append([])
    done = False
    # model.reset_action_buffer(opt.npred)
    while not done:
        policy_input_images = observation['context'].contiguous()
        policy_input_states = observation['state'].contiguous()
        action, _, _, _ = model.policy_net(policy_input_images, policy_input_states, sample=True, normalize_inputs=True,
                                           normalize_outputs=True, controls=dict(target_lanes=target_y))
        action = action.data.cpu().view(2).numpy()
        actions.append(((torch.tensor(action) - stats['a_mean']) / stats['a_std']))
        action_sequences[-1].append(action)
        state_sequences[-1].append(policy_input_states)
        observation, _, done, info = env.step(action)
        images.append(policy_input_images[-1])
        states.append(policy_input_states[-1])
        if opt.display:
            env.render()
    info.dump_state_image(os.path.join(movie_dir, 'state'))
    print('Done')

    # Save ego view
    images = torch.stack(images)
    states = torch.stack(states)
    actions = torch.stack(actions)
    costs, mu_list, std_list = None, None, None

    if len(images) > 3:
        utils.save_movie(os.path.join(movie_dir, 'ego'), images.float() / 255.0, states, costs,
                         actions=actions, mu=mu_list, std=std_list, pytorch=True)
        if opt.save_sim_video:
            sim_path = os.path.join(movie_dir, 'sim')
            print(f'[saving simulator movie to {sim_path}]')
            os.mkdir(sim_path)
            for n, img in enumerate(info.frames):
                imwrite(os.path.join(sim_path, f'im{n:05d}.png'), img)


if __name__ == '__main__':
    main()
