import torch, numpy, argparse
from dataloader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-nshards', type=int, default=2)
parser.add_argument('-T', type=int, default=10)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-lanes', type=int, default=3)
parser.add_argument('-ncond', type=int, default=4)
parser.add_argument('-npred', type=int, default=10)
parser.add_argument('-n_episodes', type=int, default=10)
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
opt = parser.parse_args()

data_file = f'{opt.data_dir}/traffic_data_lanes={opt.lanes}-episodes={opt.n_episodes}-seed=*.pkl'
print(data_file)

dataloader = DataLoader(data_file, opt)
images, states, actions = dataloader.get_batch_il('train')
inputs, actions, targets, _, _  = dataloader.get_batch_fm('train')

