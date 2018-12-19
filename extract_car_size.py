import pandas as pd
import torch

path = './traffic-data/xy-trajectories/i80/'
time_slots = (
    'trajectories-0400-0415',
    'trajectories-0500-0515',
    'trajectories-0515-0530',
)

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
        
torch.save(car_sizes, 'traffic-data/state-action-cost/data_i80_v0/car_sizes.pth')
