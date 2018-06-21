import pandas as pd
import pickle

class trj(object):

    def __init__(self, xdata=[], ydata=[]):
        self.x = xdata
        self.y = ydata

# Conversion LANE_W from real world to pixels
# A US highway lane width is 3.7 metres, here 50 pixels
LANE_W = 24  # pixels / 3.7 m, lane width
SCALE = LANE_W / 3.7  # pixels per metre
FOOT = 0.3048  # metres per foot
X_OFFSET = 470  # horizontal offset (camera 2 leftmost view)
MAX_SPEED = 130

file_name = '/home/boliu/Projects/pytorch-Traffic-Simulator/data_i80/trajectories-0400-0415.txt'

car_list = [1,100,200,400,250]

df = pd.read_table(file_name, sep='\s+', header=None, names=(
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

y_list = []
x_list = []
for car in car_list:
    y_list.append(((df.loc[df['Vehicle ID'] == car]['Local X'])*FOOT).tolist())
    x_list.append(((df.loc[df['Vehicle ID'] == car]['Local Y'])*FOOT).tolist())

for i, car in enumerate(car_list):
    print(car)
    data = trj(x_list[i], y_list[i])
    with open('trj'+str(car)+'.pickle', 'wb') as f:
        pickle.dump(data, f)


