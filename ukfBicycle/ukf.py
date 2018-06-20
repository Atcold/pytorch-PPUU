import numpy as np
from filterpy.kalman import UKF
from filterpy.kalman import MerweScaledSigmaPoints
from numba.decorators import jit
from math import tan, sin, cos, sqrt, atan2

@jit
def move(x, dt, wheelbase=2.5, u=np.array([0.,0.]), eps=1e-5):
    # state vector: (px, py, v, phi)
    hdg = x[3]
    vel = x[2]
    steering_angle = u[1]
    acc = u[0]
    dist = vel * dt + 0.5 * acc * dt**2

    if abs(steering_angle) > eps:  # is robot turning?
        beta = (dist / wheelbase) * tan(steering_angle)
        r = wheelbase / tan(steering_angle)  # radius

        sinh, sinhb = sin(hdg), sin(hdg + beta)
        cosh, coshb = cos(hdg), cos(hdg + beta)

        return x + np.array([-r * sinh + r * sinhb, r * cosh - r * coshb, acc*dt, beta])
    else:  # moving in straight line
        return x + np.array([dist * cos(hdg), dist * sin(hdg), acc * dt, 0])

@jit
def transition(x, dt, wheelbase, eps=1e-5):
    hdg = x[3]
    vel = x[2]
    steering_angle = u[1]
    dist = vel * dt

    if abs(steering_angle) > eps: # is robot turning?
        beta = (dist / wheelbase) * tan(steering_angle)
        r = wheelbase / tan(steering_angle) # radius

        sinh, sinhb = sin(hdg), sin(hdg + beta)
        cosh, coshb = cos(hdg), cos(hdg + beta)

        if x[3] + beta > np.pi or x[3] + beta < -np.pi:
            print('transition phi out of range')

        return x + np.array([-r*sinh + r*sinhb,
                              r*cosh - r*coshb, beta])
    else: # moving in straight line
        return x + np.array([dist*cos(hdg), dist*sin(hdg), 0])

@jit
def normalize_angle(x):
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > 0.5*np.pi:          # move to [-pi, pi)
        x -= 2 * np.pi
    return x


# @jit
# def residual_h(a, b):
#     # measurement is (position_x, position_y)
#     z = a - b
#     return z

@jit
def residual_x(a, b):
    x = a - b
    # state vector is (x, y, v, phi)
    x[3] = normalize_angle(x[3])
    return x

@jit
def Hx(x):
    """ takes a state variable and returns the measurement
    that would correspond to that state. """

    return np.array([x[0], x[1]])


@jit
def state_mean(sigmas, Wm):
    x = np.zeros(4)

    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
    x[2] = np.sum(np.dot(sigmas[:, 2], Wm))

    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 3]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 3]), Wm))
    x[3] = atan2(sum_sin, sum_cos)
    return x

# @jit
# def z_mean(sigmas, Wm):
#     z = np.zeros([0,0])
#     z[0] = np.sum(np.dot(sigmas[:, 0], Wm))
#     z[1] = np.sum(np.dot(sigmas[:, 1], Wm))
#
#     return z


class ukf(object):

    def __init__(self):
        # state vector: [px, py, phi]
        self.dt = 0.1
        self.points = MerweScaledSigmaPoints(n=4, alpha=.00001, beta=2, kappa=1, subtract=residual_x)
        self.ukf = UKF.UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=self.dt, fx=move, hx=Hx, points=self.points,
                                             x_mean_fn=state_mean, residual_x=residual_x)
        self.ukf.x = np.array([0.,0.,0.,0.])
        #self.ukf.P = np.diag([.1, .1, .1, .05])
        self.ukf.P = np.array([[1, 0, 1, 0],
                               [0, 1, 1, 0],
                               [1, 1, 2, 0],
                               [0, 0, 0, 1]])

        self.ukf.R = np.diag([0.5**2, 0.25**2])
        self.ukf.Q = np.eye(4)*0.001

    def predict(self):
        self.ukf.predict()

    def update(self, meas):
        self.ukf.update(meas)
