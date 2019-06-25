import argparse
import os

import pandas as pd
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-map', type=str, default='i80', choices={'ai', 'i80', 'us101', 'lanker', 'peach'})
opt = parser.parse_args()

path = './traffic-data/xy-trajectories/{}/'.format(opt.map)
trajectories_path = './traffic-data/state-action-cost/data_{}_v0'.format(opt.map)
time_slots = [d[0].split("/")[-1] for d in os.walk(trajectories_path) if d[0] != trajectories_path]

df = dict()
for ts in time_slots:
    df[ts] = pd.read_table(path + ts + '.txt', sep='\s+', header=None, names=(
        'Vehicle ID',
        'Frame ID',
        'Total Frames',
        'Global Time',
        'Local X',
        'Local Y',
        'Global X',
        'Global Y',
        'Vehicle Length',
        'Vehicle Width',
        'Vehicle Class',
        'Vehicle Velocity',
        'Vehicle Acceleration',
        'Lane Identification',
        'Preceding Vehicle',
        'Following Vehicle',
        'Spacing',
        'Headway'
    ))

car_sizes = dict()
for ts in time_slots:
    d = df[ts]
    car = lambda i: d[d['Vehicle ID'] == i]
    car_sizes[ts] = dict()
    cars = set(d['Vehicle ID'])
    for c in cars:
        if len(car(c)) > 0:
            size = tuple(car(c).loc[car(c).index[0], ['Vehicle Width', 'Vehicle Length']].values)
            car_sizes[ts][c] = size
            print(c)

torch.save(car_sizes, 'traffic-data/state-action-cost/data_{}_v0/car_sizes.pth'.format(opt.map))
