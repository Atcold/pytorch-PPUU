from dataloader import DataLoader
import torch
from collections import namedtuple
import pickle
import utils

print('> Loading DataLoader')
class opt:
    debug = 0
dataloader = DataLoader(None, opt, 'i80')

print('> Loading splits')
splits = torch.load('/home/atcold/vLecunGroup/nvidia-collab/traffic-data-atcold/data_i80_v0/splits.pth')

for split in splits:
    data_dict = dict()
    print(f'> Building {split}')
    for idx in splits[split]:
        car_path = dataloader.ids[idx]
        timeslot, car_id = utils.parse_car_path(car_path)
        data_dict[idx] = timeslot, car_id
    print(f'> Pickling {split}')
    with open(f'{split}.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
