import numpy as np
from filterpy.kalman import UKF
from filterpy.kalman import MerweScaledSigmaPoints
from numba.decorators import jit
from math import tan, sin, cos, sqrt, atan2
import copy

@jit
def move(x, dt, wheelbase=2.5, u=np.array([0.,0.]), eps=1e-5):
    """
    Bicyle motion model for a Ukf.
    :param x: state vector [px (m), py (m), speed (m/s), hdg (rad)]
    :param dt: dt
    :param wheelbase: axle distance of a two-axle vehicle
    :param u: control command if issued: [acceleration (m/s**2), steering angle (rad)]
    :param eps: is used for small number test
    :return: predicted state vector
    """
    # state vector: (px, py, speed, phi)
    hdg = x[3]
    speed = x[2]
    steering_angle = u[1]
    acc = u[0]
    distance_traveled = speed * dt + 0.5 * acc * dt**2
    ret = np.array([0.,0.,0.,0.])
    if abs(steering_angle) > eps and speed > eps * 1e5:  # non-zero steering: turning
        # beta = change of heading (rad)
        beta = (distance_traveled / wheelbase) * tan(steering_angle)
        # r = turn radius w.r.t. rear axle
        r = wheelbase / tan(steering_angle)

        sinh, sinhb = sin(hdg), sin(hdg + beta)
        cosh, coshb = cos(hdg), cos(hdg + beta)
        ret = x + np.array([-r * sinh + r * sinhb, r * cosh - r * coshb, acc*dt, beta])
    else:  # moving in a straight line
        ret = x + np.array([distance_traveled * cos(hdg), distance_traveled * sin(hdg), acc * dt, 0])

    return ret

@jit
def normalize_angle(hdg):
    """
    normalize hdg radian number to [-PI, PI]
    :param hdg: an angle
    :return: normalized angle
    """
    hdg = hdg % (2 * np.pi)
    if hdg > 0.5*np.pi:
        hdg -= 2 * np.pi
    return hdg


@jit
def residual_x(a, b):
    x = a - b
    # state vector is (x, y, speed, hdg)
    x[3] = normalize_angle(x[3])
    return x

@jit
def Hx(x):
    """
    measurement function: maps to measurement space: (px, py)
    :param x: state vector
    :return: state vector in measurement space
    """
    return np.array([x[0], x[1]])


@jit
def state_mean(sigmas, Wm):
    """
    functor for the Ukf to estimate mean of state vector
    :param sigmas: sigma points
    :param Wm: weights
    :return: state mean estimate
    """
    x = np.zeros(4)

    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
    x[2] = np.sum(np.dot(sigmas[:, 2], Wm))

    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 3]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 3]), Wm))
    x[3] = atan2(sum_sin, sum_cos)
    return x

@jit
def state_mean_iterative(sigmas, Wm):
    """
    functor for the Ukf to estimate mean of state vector
    :param sigmas: sigma points
    :param Wm: weights
    :return: state mean estimate
    """
    x = np.zeros(4)

    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
    x[2] = np.sum(np.dot(sigmas[:, 2], Wm))
    WmCopy = copy.deepcopy(Wm)
    sigmasCopy = copy.deepcopy(sigmas)
    for i in range(1, len(Wm)):
        WmSum = WmCopy[i-1] + WmCopy[i]
        rollingAvg = (WmCopy[i-1]/WmSum) * sigmasCopy[i-1,3] + (WmCopy[i]/WmSum) * sigmasCopy[i,3]
        WmCopy[i-1] = WmSum
        sigmasCopy[i-1,3] = normalize_angle(rollingAvg)

    x[3] = normalize_angle(sigmasCopy[len(Wm)-2, 3])
    return x


class Ukf(object):
    """
    Unscented Kalman Filter with augmented bicycle motion model.
    state vector: [position_x (m), position_y (m), vehicle_speed (m/s), vehicle_heading (rad)]
    control cmd: [acceleration (m/s**2), steering angle (rad)]
    """

    def __init__(self, dt=0.1, wheelbase=2.5, startx=0., starty=0., startspeed=0., starthdg=0., stdx=1.0, stdy=1.0, noise=0.02):
        """
        initialize Ukf
        :param dt: dt
        :param wheelbase: axle distance of a two-axle vehicle (default = 2.5m)
        :param startx: initial position x (default = 0m)
        :param starty: initial position y (default = 0m)
        :param startspeed: initial speed (default = 0m/s)
        :param starthdg: initial heading (default = 0 rad, x-axis aligned)
        :param stdx: std of measurement for position x (default = 1.0m)
        :param stdy: std of measurement for position y (default = 1.0m)
        :param noise: process noise for motion model (default = 0.1m)
        """
        # state vector: [px, py, speed, hdg (rad)]
        self.dt = dt
        self.x = np.array([startx, starty, startspeed, starthdg])
        self.wheelbase = wheelbase
        self.points = MerweScaledSigmaPoints(n=4, alpha=.00001, beta=2, kappa=1, subtract=residual_x)
        self.ukf = UKF.UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=self.dt, fx=move, hx=Hx, points=self.points,
                                             x_mean_fn=state_mean_iterative, residual_x=residual_x)
        self.ukf.x = np.array([startx, starty, startspeed, starthdg])
        # initial guess
        self.ukf.P = np.array([[1, 0, 1, 0],
                               [0, 1, 1, 0],
                               [1, 1, 2, 0],
                               [0, 0, 0, 1]])

        # estimated from data
        self.ukf.P = np.array([[ 4.78559217e-01, -1.94838441e-03,  2.48955909e-01, -1.25313018e-03],
                               [-1.94838441e-03,  7.63804713e-01,  1.69689503e-03,  1.83459578e-01],
                               [ 2.48955909e-01,  1.69689503e-03,  1.51635257e+00,  6.06738360e-06],
                               [-1.25313018e-03,  1.83459578e-01,  6.06738360e-06,  2.18001626e-01]], dtype=np.float128)

        self.ukf.R = np.diag([stdx**2, stdy**2])
        self.ukf.Q = np.eye(4, dtype=np.float128)*noise

    def predict(self, action=np.array([0.,0.])):
        """
        prediction step of Ukf using a motion model
        :return:
        """
        self.ukf.predict(wheelbase=self.wheelbase, u=action)

    def update(self, meas):
        """
        update Ukf with a measurement
        :param meas: measurement vector [px, py]
        :return: void
        """
        self.ukf.update(meas)

    def step(self, meas, u=np.array([0.,0.])):
        """
        client code interface to update Ukf with a new measurement
        :param meas: measurement vector [px, py]
        :return: void
        """
        self.predict(action=u)
        self.update(meas)

    def get_position(self):
        return np.array([self.ukf.x[0], self.ukf.x[1]])

    def get_heading(self):
        return self.ukf.x[3]

    def get_speed(self):
        return self.ukf.x[2]
