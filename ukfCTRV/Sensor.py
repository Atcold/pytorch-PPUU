import numpy as np
from .ukf import ukf

sensor_types = {'I80': 0, 'GPS': 1, 'IMU': 2, 'Lidar': 3}


class Sensor(object):
    """
    virtual class: Sensor interface to be used with a UKF
        a Sensor object requires to reference a UKF object for sensor fusion
    """

    def __init__(self, name='', active=False, noise=None, sensor_type=None, ukf=None, data_dimension=None):
        self.name = name
        self.is_active = active
        self._noise = noise
        self.R = np.diag(self._noise)
        self.sensor_type = sensor_type
        self._ukf = ukf
        self._dim = data_dimension
        self.initialized = False
        self.has_new_data = False
        self.measurement = None

    @property
    def ukf(self):
        return self._ukf

    @ukf.setter
    def ukf(self, val):
        self._ukf = val

    @ukf.deleter
    def ukf(self):
        self._ukf = None

    @property
    def noise(self):
        return self._noise

    @noise.setter
    def noise(self, val):
        self._noise = val

    @noise.deleter
    def noise(self):
        self._noise = None

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, val):
        self._dim = val

    def enable(self):
        self.is_active = True

    def disable(self):
        self.is_active = False

    def update_measurement(self, meas):
        pass

    def update_ukf(self):
        pass


class I80measurement(Sensor):

    def __init__(self, name, stdx=0.1, stdy=0.2):
        super(I80measurement, self).__init__(
            name=name, active=True, noise=np.asarray([stdx, stdy]),
            sensor_type=sensor_types['I80'], data_dimension=2)
        self.measurement = np.asanyarray([0., 0.])

    def update_measurement(self, meas):
        if self.is_active:
            self.measurement = meas
            if not self.initialized:
                self.initialized = True
            self.has_new_data = True

    def update_ukf(self):
        if not self.has_new_data or not self.initialized:
            return

        # update and sensor fusion
        n_x = self._ukf.n_x
        n_aug = n_x + 2
        n_z = self._dim
        weights = self._ukf.weights
        Xsig_pred = self._ukf.Xsig_pts
        x = self._ukf.x
        P = self._ukf.P

        # determine sigma points in the measurement space
        Zsig = np.zeros([n_z, 2*n_aug + 1])

        # transform sigma points into measurement space
        for i in range(2*n_aug+1):
            # extract values for better readability
            px = Xsig_pred[0,i]
            py = Xsig_pred[1,i]

            Zsig[0,i] = px
            Zsig[1,i] = py

        # recover mean and cov for predicted measurement
        z_pred = np.zeros(n_z)
        for i in range(2*n_aug+1):
            z_pred = z_pred + weights[i] * Zsig[:,i]

        # covariance matrix S
        S = np.zeros([n_z, n_z])
        for i in range(2*n_aug+1):
            z_diff = Zsig[:,i] - z_pred
            S = S + weights[i]*np.outer(z_diff, np.transpose(z_diff))

        S = S + self.R

        # update state: x and P
        # create matrix for cross correlation Tc
        Tc = np.zeros([n_x, n_z])
        for i in range(2*n_aug+1):
            z_diff = Zsig[:,i] - z_pred
            x_diff = Xsig_pred[:,i] - x

            Tc = Tc + weights[i]*np.outer(x_diff, np.transpose(z_diff))

        # Kalman gain K
        K = np.matmul(Tc, np.linalg.inv(S))

        z_diff = self.measurement - z_pred

        x = x + np.matmul(K, z_diff)
        P = P - np.matmul(np.matmul(K, S), np.transpose(K))

        self._ukf.x = x
        self._ukf.P = P

        self.has_new_data = False




