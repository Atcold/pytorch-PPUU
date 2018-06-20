import matplotlib.pyplot as plt
import pickle
from .ukf import *

class trj(object):

    def __init__(self, xdata=[], ydata=[]):
        self.x = xdata
        self.y = ydata

car_list = [1,100,200,400,250]
car_data = []

for i, car in enumerate(car_list):
    with open('trj'+str(car)+'.pickle', 'rb') as f:
        car_data.append(pickle.load(f))

fig = plt.figure(figsize=(20, 4))
plt.subplot(len(car_list),1,1)

for i, car in enumerate(car_list):
    plt.subplot(len(car_list), 1, i+1)
    plt.scatter(car_data[i].x, car_data[i].y,s=1)
    plt.title('car ' + str(car))

plt.show()
