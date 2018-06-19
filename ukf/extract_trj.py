import pandas as pd
import matplotlib.pyplot as plt
import pickle

file_name = '/home/boliu/Projects/pytorch-Traffic-Simulator/data_i80/trajectories-0400-0415.txt'
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

y = df.loc[df['Vehicle ID'] == 2]['Local X'].tolist()
x = df.loc[df['Vehicle ID'] == 2]['Local Y'].tolist()

print(len(x))

z = zip(x, y)


class trj(object):

    def __init__(self, xdata=[], ydata=[]):
        self.x = xdata
        self.y = ydata

data = trj(x,y)

with open('trj2.pickle', 'wb') as f:
    pickle.dump(data, f)

with open('trj2.pickle', 'rb') as f:
    data = pickle.load(f)

plt.scatter(data.x, data.y)
plt.show()