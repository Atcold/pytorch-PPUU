import pickle
import numpy as np
from ukfBicycle import ukf
import matplotlib.pyplot as plt
from numba.decorators import jit


@jit
def plot_phi(x, y, phi, length):
    # find the end point
    endy = y + length * np.sin(phi)
    endx = x + length * np.cos(phi)

    plt.plot([x, endx], [y, endy], 'g-')

class trj(object):

    def __init__(self, xdata=[], ydata=[]):
        self.x = xdata
        self.y = ydata


car_list = [1,100,200,400,250]
car_data = []
for i, car in enumerate(car_list):
    with open('/home/boliu/Projects/pytorch-Traffic-Simulator/test_KF/ScrData/trj'+str(car)+'.pickle', 'rb') as f:
        car_data.append(pickle.load(f))


ukf_car = []
for car in car_list:
    ukf_car.append(ukf.ukf())


px_kf_list = []
py_kf_list = []
phi_kf_list = []
for i, car in enumerate(car_list):
    print(i)
    px_kf = []
    py_kf = []
    phi_kf = []
    for px, py in zip(car_data[i].x, car_data[i].y):
        ukf_car[i].predict()
        z = np.array([px, py])
        ukf_car[i].update(z)
        px_kf.append(ukf_car[i].ukf.x[0])
        py_kf.append(ukf_car[i].ukf.x[1])
        phi_kf.append(ukf_car[i].ukf.x[2])
    px_kf_list.append(px_kf)
    py_kf_list.append(py_kf)
    phi_kf_list.append(phi_kf)

fig = plt.figure(figsize=(20, 4))
plt.subplot(len(car_list),1,1)

for i, car in enumerate(car_list):
    ax = plt.subplot(len(car_list), 1, i+1)
    plt.scatter(car_data[i].x, car_data[i].y, s=1)
    plt.plot(px_kf_list[i], py_kf_list[i], 'r-')
    for px,py, phi in zip(px_kf_list[i], py_kf_list[i], phi_kf_list[i]):
        plot_phi(px, py, phi, 1)
    plt.ylabel('car ' + str(car))
    ax.yaxis.set_label_position("right")

plt.show()
exit()
