import numpy as np
from filterpy.kalman import UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
from numba.decorators import jit


@jit
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

    # phi normalization
    while x[3] > np.pi:
        x[3] = x[3] - 2.*np.pi
    while x[3] < -np.pi:
        x[3] = x[3] + 2.*np.pi

    return x_prior


@jit
def hx(x):
    """
    measurement function: maps state to measurement space
    :param x: state vector = [px, py, v, phi, phi_dot]
    :return: z measurement space values = [px, py]
    """
    return np.array([x[0], x[1]])

@jit
def state_mean(sigmas, Wm):
    x = np.zeros(5)

    x[0] = np.sum(np.dot(sigmas[:,0], Wm))
    x[1] = np.sum(np.dot(sigmas[:,1], Wm))
    x[2] = np.sum(np.dot(sigmas[:,2], Wm))
    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 3]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 3]), Wm))
    x[3] = np.atan2(sum_sin, sum_cos)
    x[4] = np.sum(np.dot(sigmas[:,4], Wm))

    return x


@jit
def plot_phi(x, y, phi, length):
    # find the end point
    endy = y + length * np.sin(phi)
    endx = x + length * np.cos(phi)

    plt.plot([x, endx], [y, endy], 'g-')


class ukfCTRV(object):

    def __init__(self):
        self.x = np.array([0,0,0,0,0])
        self.P = np.diag([10,1,10,1,10])
        self.zx_std = 1.0
        self.zy_std = 0.5
        self.R = np.diag([self.zx_std**2, self.zy_std**2])
        self.Q = np.array([[10, 0, 0,   0,   0],
                  [0, 10, 0,   0,   0],
                  [0,  0, 100, 25,   10],
                  [0,  0, 25,  10,   0],
                  [0,  0, 10,   0, 100]])
        self.dt = 0.1
        self.points = MerweScaledSigmaPoints(n=5, alpha=1, beta=2, kappa=3-5)
        self.ukf = UKF.UnscentedKalmanFilter(dim_x=len(self.x), dim_z=self.R.shape[0],
                                             dt=self.dt, fx=fx, hx=hx, points=self.points)
        self.ukf.R = self.R
        self.ukf.Q = self.Q

    def predict(self):
        self.ukf.predict()
        self.x = self.ukf.x

    def update(self, meas):
        self.ukf.update(meas)
        self.x = self.ukf.x