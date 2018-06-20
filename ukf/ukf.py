import numpy as np
from filterpy.kalman import UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
import pickle


class trj(object):

    def __init__(self, xdata=[], ydata=[]):
        self.x = xdata
        self.y = ydata


def fx(x, dt, eps=1e-5):
    """
    process model: x = [px, py, v, phi, phi_dot]
    :return: prior x_prior prediction
    """
    if abs(x[3]) < eps:
        x_prior = x + np.array([x[2]*np.cos(x[3])*dt, x[2]*np.sin(x[3])*dt, 0., x[4]*dt, 0.])
    else:
        x_prior = x + np.array([
            (x[2]/x[3]) * (np.sin(x[3]+x[4]*dt) - np.sin(x[3])),
            (x[2]/x[3]) * (-np.cos(x[3]+x[4]*dt) + np.cos(x[3])),
            0,
            x[4]*dt,
            0
        ])

    return x_prior


def hx(x):
    """
    measurement function: maps state to measurement space
    :param x: state vector = [px, py, v, phi, phi_dot]
    :return: z measurement space values = [px, py]
    """
    return np.array([x[0], x[1]])


dt = 0.1
points = MerweScaledSigmaPoints(n=5, alpha=1, beta=2, kappa=3-5)
ukf = UKF.UnscentedKalmanFilter(dim_x=5, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)

ukf.x = np.array([0,0,0,0,0])
ukf.P *= np.diag([10,1,10,1,10])

zx_std = 1.0
zy_std = 0.5

ukf.R = np.diag([zx_std**2, zy_std**2])
#ukf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=1**2, block_size=2)

# state vector = [px, py, v, phi, phi_dot]
# ukf.Q = np.array([[1, 5.0e-06, 0.0e+00, 0.0e+00, 0e+00],
#                  [5.0e-06, 1, 0.0e+00, 0.0e+00, 0e+00],
#                  [0.0e+00, 0.0e+00, 1, 5.0e-06, 4.0e-06],
#                  [0.0e+00, 0.0e+00, 5.0e-06, 1, 1.0e-04],
#                  [0.0e+00, 0.0e+00, 4.0e-06, 1.0e-04, 1]])

ukf_car = [ukf]*5

car_list = [1,100,200,400,250]
car_data = []

for i, car in enumerate(car_list):
    with open('trj'+str(car)+'.pickle', 'rb') as f:
        car_data.append(pickle.load(f))

px_kf_list = []
py_kf_list = []
for i, car in enumerate(car_list):
    px_kf = []
    py_kf = []
    for px, py in zip(car_data[i].x, car_data[i].y):
        ukf_car[i].predict()
        z = np.array([px, py])
        ukf_car[i].update(z)
        px_kf.append(ukf_car[i].x[0])
        py_kf.append(ukf_car[i].x[1])
    px_kf_list.append(px_kf)
    py_kf_list.append(py_kf)


fig = plt.figure(figsize=(20, 4))
plt.subplot(len(car_list),1,1)

for i, car in enumerate(car_list):
    ax = plt.subplot(len(car_list), 1, i+1)
    plt.scatter(car_data[i].x, car_data[i].y, s=1)
    plt.plot(px_kf_list[i], py_kf_list[i], 'r-')
    plt.ylabel('car ' + str(car))
    ax.yaxis.set_label_position("right")

plt.show()


### single ukf testing
x = car_data[4].x
y = car_data[4].y
plt.scatter(x, y, s=1)

dt = 0.1
points = MerweScaledSigmaPoints(n=5, alpha=1, beta=2, kappa=3-5)
ukf = UKF.UnscentedKalmanFilter(dim_x=5, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)

ukf.x = np.array([0,0,0,np.pi/2,0])
ukf.P *= 0.2

zx_std = 1.0
zy_std = 1.5

ukf.R = np.diag([zx_std**2, zy_std**2])
ukf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=1**2, block_size=2)

ukf.Q = np.array([[2.5e-04, 5.0e-06, 0.0e+00, 0.0e+00, 0e+00],
                  [5.0e-06, 1.0e-04, 0.0e+00, 0.0e+00, 0e+00],
                  [0.0e+00, 0.0e+00, 2.5e-07, 5.0e-06, 4.0e-06],
                  [0.0e+00, 0.0e+00, 5.0e-06, 1.0e-04, 1.0e-04],
                  [0.0e+00, 0.0e+00, 4.0e-06, 1.0e-04, 1.0e-04]])

px_list = []
py_list = []
for px, py in zip(x,y):
    ukf.predict()
    z = np.array([px, py])
    ukf.update(z)
    px_list.append(ukf.x[0])
    py_list.append(ukf.x[1])

plt.plot(px_list, py_list, 'r-')
plt.show()
