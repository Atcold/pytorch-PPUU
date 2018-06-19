import numpy as np
from filterpy.kalman import UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
import pickle


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


class trj(object):

    def __init__(self, xdata=[], ydata=[]):
        self.x = xdata
        self.y = ydata

dt = 0.1
points = MerweScaledSigmaPoints(n=5, alpha=1, beta=2, kappa=3-5)
ukf = UKF.UnscentedKalmanFilter(dim_x=5, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)

ukf.x = np.array([0,0,0,np.pi/2,0])
ukf.P *= 0.2

print(ukf.P)

zx_std = 1.0
zy_std = 1.5

ukf.R = np.diag([zx_std**2, zy_std**2])
ukf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=1**2, block_size=2)

ukf.Q = np.array([[2.5e-04, 5.0e-06, 0.0e+00, 0.0e+00, 0e+00],
                  [5.0e-06, 1.0e-04, 0.0e+00, 0.0e+00, 0e+00],
                  [0.0e+00, 0.0e+00, 2.5e-07, 5.0e-06, 4.0e-06],
                  [0.0e+00, 0.0e+00, 5.0e-06, 1.0e-04, 1.0e-04],
                  [0.0e+00, 0.0e+00, 4.0e-06, 1.0e-04, 1.0e-04]])

with open('trj2.pickle', 'rb') as f:
    data = pickle.load(f)

x=data.x
y=data.y

plt.scatter(x, y)

px_list = []
py_list = []
for px, py in zip(x,y):
    ukf.predict()
    z = np.array([px, py])
    ukf.update(z)
    px_list.append(ukf.x[0])
    py_list.append(ukf.x[1])


plt.plot(px_list, py_list)
print(len(px_list))
plt.show()

print(ukf.P)