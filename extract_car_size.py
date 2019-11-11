import argparse
import os

import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-map', type=str, default='i80', choices={'ai', 'i80', 'us101', 'lanker', 'peach', 'highD'})
opt = parser.parse_args()

path = f'./traffic-data/xy-trajectories/{opt.map}/'
trajectories_path = f'./traffic-data/state-action-cost/data_{opt.map}_v0'
_, time_slots, _ = next(os.walk(trajectories_path))

df = dict()
if opt.map == 'highD':
    # Track dataframes
    dtypes_dict = {
        'Frame ID': np.int64,
        'Vehicle ID': np.int64,
        'Local X': np.float64,
        'Local Y': np.float64,
        'Vehicle Length': np.float64,
        'Vehicle Width': np.float64,
        'Vehicle Velocity X': np.float64,
        'Vehicle Velocity Y': np.float64,
        'Vehicle Acceleration X': np.float64,
        'Vehicle Acceleration Y': np.float64,
        'Front Sight Distance': np.float64,
        'Back Sight Distance': np.float64,
        'Spacing': np.float64,
        'Headway': np.float64,
        'Time to Collision': np.float64,
        'Preceding Velocity X': np.float64,
        'Preceding Vehicle': np.int64,
        'Following Vehicle': np.int64,
        'Left Preceding ID': np.int64,
        'Left Alongside ID': np.int64,
        'Left Following ID': np.int64,
        'Right Preceding ID': np.int64,
        'Right Alongside ID': np.int64,
        'Right Following ID': np.int64,
        'Lane Identification': np.int64,
    }
    for ts in time_slots:
        df[ts] = pd.read_csv(os.path.join(path, f'{ts}_tracks.csv'),
                             header=0,
                             names=(
                                 'Frame ID',
                                 'Vehicle ID',
                                 'Local X',
                                 'Local Y',
                                 'Vehicle Length',
                                 'Vehicle Width',
                                 'Vehicle Velocity',
                                 'Vehicle Velocity Y',
                                 'Vehicle Acceleration',
                                 'Vehicle Acceleration Y',
                                 'Front Sight Distance',
                                 'Back Sight Distance',
                                 'Spacing',
                                 'Headway',
                                 'Time to Collision',
                                 'Preceding Velocity X',
                                 'Preceding Vehicle',
                                 'Following Vehicle',
                                 'Left Preceding ID',
                                 'Left Alongside ID',
                                 'Left Following ID',
                                 'Right Preceding ID',
                                 'Right Alongside ID',
                                 'Right Following ID',
                                 'Lane Identification'
                             ),
                             dtype=dtypes_dict)
else:
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
    def car(i): return d[d['Vehicle ID'] == i]
    car_sizes[ts] = dict()
    cars = set(d['Vehicle ID'])
    for c in cars:
        if len(car(c)) > 0:
            size = tuple(car(c).loc[car(c).index[0], ['Vehicle Width', 'Vehicle Length']].values)
            car_sizes[ts][c] = size
            print(c)

torch.save(car_sizes, f'traffic-data/state-action-cost/data_{opt.map}_v0/car_sizes.pth')

