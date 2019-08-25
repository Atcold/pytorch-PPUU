import argparse
import os

import pandas as pd
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-map', type=str, default='i80', choices={'ai', 'i80', 'us101', 'lanker', 'peach'})
parser.add_argument('-v', type=int, default=0)
opt = parser.parse_args()


path = f'./traffic-data/xy-trajectories/{opt.map}/'
trajectories_path = f'./traffic-data/state-action-cost/data_{opt.map}_v{opt.v}'
_, time_slots, _ = next(os.walk(trajectories_path))

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

torch.save(car_sizes, f'traffic-data/state-action-cost/data_{opt.map}_v{opt.v}/car_sizes.pth')
