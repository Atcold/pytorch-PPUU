import torch, numpy, argparse
from dataloader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-nshards', type=int, default=1)
parser.add_argument('-T', type=int, default=20)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-lanes', type=int, default=8)
parser.add_argument('-ncond', type=int, default=10)
parser.add_argument('-npred', type=int, default=20)
parser.add_argument('-n_episodes', type=int, default=20)
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
opt = parser.parse_args()


data_file = f'{opt.data_dir}/traffic_data_lanes={opt.lanes}-episodes=*-seed=*.pkl'
print(data_file)

dataloader = DataLoader(data_file, opt, dataset='i80')
for i in range(2):
    inputs, actions, targets, _, _ = dataloader.get_batch_fm('train')
#for _ in range(10):
#    inputs, actions, targets, _, _  = dataloader.get_batch_fm('train')

