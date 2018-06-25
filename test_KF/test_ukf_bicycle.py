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
    """
    this is the data type stored in pickle
    """

    def __init__(self, xdata=[], ydata=[]):
        self.x = xdata
        self.y = ydata


# from data_i80/trajectories-0400-0415
car_list = [1,100,200,400,250]
car_list = [1]
car_data = []
# for i, car in enumerate(car_list):
#     with open('ScrData/trj'+str(car)+'.pickle', 'rb') as f:
#         car_data.append(pickle.load(f))

with open('../v35.pickle', 'rb') as f: trajectory = pickle.load(f)
car_data.append(trj(trajectory[:, 0], trajectory[:, 1]))

ukf_car = []
for i, car in enumerate(car_list):
    px, py = car_data[i].x[0], car_data[i].y[0]
    ukf_car.append(ukf.Ukf(dt=0.1, startx=px, starty=py, noise=1e-6))

px_kf_list = []
py_kf_list = []
speed_kf_list = []
phi_kf_list = []
for i, car in enumerate(car_list):
    print("processing car # {}".format(car))
    px_kf = []
    py_kf = []
    speed_kf = []
    phi_kf = []
    for px, py in zip(car_data[i].x, car_data[i].y):
        z = np.array([px, py])
        ukf_car[i].step(z)
        # print(z, ukf_car[i].ukf.x)
        px_kf.append(ukf_car[i].ukf.x[0])
        py_kf.append(ukf_car[i].ukf.x[1])
        speed_kf.append(ukf_car[i].ukf.x[2])
        phi_kf.append(ukf_car[i].ukf.x[3])
    px_kf_list.append(px_kf)
    py_kf_list.append(py_kf)
    speed_kf_list.append(speed_kf)
    phi_kf_list.append(phi_kf)

fig = plt.figure(figsize=(20, 4))

for i, car in enumerate(car_list):
    ax = plt.subplot(len(car_list), 1, i+1)
    plt.scatter(car_data[i].x, car_data[i].y, s=10,)
    plt.plot(px_kf_list[i], py_kf_list[i], 'r-')
    for px,py,speed,phi in zip(px_kf_list[i], py_kf_list[i], speed_kf_list[i], phi_kf_list[i]):
        plot_phi(px, py, phi, speed*0.5)
    plt.ylabel('car ' + str(car))
    ax.yaxis.set_label_position("right")

plt.show()
exit()
