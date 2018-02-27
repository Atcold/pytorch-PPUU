import torch, numpy, argparse
from dataloader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-T', type=int, default=10)
parser.add_argument('-batch_size', type=int, default=3)
parser.add_argument('-lanes', type=int, default=3)
parser.add_argument('-n_episodes', type=int, default=10)
parser.add_argument('-data_dir', type=str, default='data/')
opt = parser.parse_args()

data_file = f'{opt.data_dir}/traffic_data_lanes={opt.lanes}-episodes={opt.n_episodes}-seed={opt.seed}.pkl'

dataloader = DataLoader(data_file, opt)
states, masks, actions = dataloader.get_batch('train')

