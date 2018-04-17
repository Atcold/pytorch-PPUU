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
parser.add_argument('-debug', type=int, default=0)

opt = parser.parse_args()


data_file = f'{opt.data_dir}/traffic_data_lanes={opt.lanes}-episodes=*-seed=*.pkl'
print(data_file)

dataloader = DataLoader(None, opt, dataset='i80')
for i in range(2):
    inputs, actions, targets = dataloader.get_batch_fm('train')

