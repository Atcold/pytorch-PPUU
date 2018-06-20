import pickle
import numpy as np
from ukf import ukf
import matplotlib.pyplot as plt


class trj(object):

    def __init__(self, xdata=[], ydata=[]):
        self.x = xdata
        self.y = ydata


car_list = [1,100,200,400,250]
car_data = []
for i, car in enumerate(car_list):
    with open('test_KF/ScrData/trj'+str(car)+'.pickle', 'rb') as f:
        car_data.append(pickle.load(f))


ukf_car = []
for car in car_list:
    ukf_car.append(ukf.ukfCTRV())


px_kf_list = []
py_kf_list = []
v_kf_list = []
phi_kf_list = []
for i, car in enumerate(car_list):
    print(i)
    px_kf = []
    py_kf = []
    v_kf = []
    phi_kf = []
    for px, py in zip(car_data[i].x, car_data[i].y):
        ukf_car[i].predict()
        z = np.array([px, py])
        ukf_car[i].update(z)
        px_kf.append(ukf_car[i].x[0])
        py_kf.append(ukf_car[i].x[1])
        v_kf.append(ukf_car[i].x[2])
        phi_kf.append(ukf_car[i].x[3])
    px_kf_list.append(px_kf)
    py_kf_list.append(py_kf)
    v_kf_list.append(v_kf)
    phi_kf_list.append(phi_kf)

fig = plt.figure(figsize=(20, 4))
plt.subplot(len(car_list),1,1)

for i, car in enumerate(car_list):
    ax = plt.subplot(len(car_list), 1, i+1)
    plt.scatter(car_data[i].x, car_data[i].y, s=1)
    plt.plot(px_kf_list[i], py_kf_list[i], 'r-')
    for px,py,v, phi in zip(px_kf_list[i], py_kf_list[i],v_kf_list[i], phi_kf_list[i]):
        ukf.plot_phi(px, py, phi, v*0.1)
    plt.ylabel('car ' + str(car))
    ax.yaxis.set_label_position("right")

plt.show()
exit()
