import numpy as np
from filterpy.kalman import UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
import pickle
from numba.decorators import jit


class trj(object):

    def __init__(self, xdata=[], ydata=[]):
        self.x = xdata
        self.y = ydata

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


car_list = [1,100,200,400,250]
car_data = []
for i, car in enumerate(car_list):
    with open('trj'+str(car)+'.pickle', 'rb') as f:
        car_data.append(pickle.load(f))


ukf_car = []
for car in car_list:
    ukf_car.append(ukfCTRV())


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
        print(z)
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
        plot_phi(px, py, phi, v*0.1)
    plt.ylabel('car ' + str(car))
    ax.yaxis.set_label_position("right")

plt.show()
exit()

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

ukf.Q = np.array([[10, 0, 0,   0,   0],
                  [0, 10, 0,   0,   0],
                  [0,  0, 100, 25,   10],
                  [0,  0, 25,  10,   0],
                  [0,  0, 10,   0, 100]])

px_list = []
py_list = []
v_list = []
phi_list = []
for px, py in zip(x,y):
    ukf.predict()
    z = np.array([px, py])
    ukf.update(z)
    px_list.append(ukf.x[0])
    py_list.append(ukf.x[1])
    v_list.append(ukf.x[2])
    phi_list.append(ukf.x[3])


plt.plot(px_list, py_list, 'r-')
for px,py,v, phi in zip(px_list, py_list, v_list, phi_list):
    plot_phi(px, py, phi, v*0.2)

plt.show()
