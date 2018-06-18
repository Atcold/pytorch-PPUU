import numpy as np
from sklearn import datasets

class ukf(object):

    def __init__(self):
        # sensor list for fusion algorithm
        self._sensors = None
        # state dimension
        self.n_x = 5
        # px, py, v, phi, phi_dot
        self.x = np.zeros(self.n_x)

        self.time = 0.
        self.dt = 1. # hardcoded for TrafficSim

        # noise for acceleration, needs to be estimated
        self.std_a = 0.75
        # noise for yaw acceleration, needs to be estimated
        self.std_yawdd = 0.5

        # process covariance, needs a better initialized value
        self.P = np.identity(self.n_x)

        # sigma points spreading parameter
        n_aug = self.n_x + 2
        self.lamba = 3 - n_aug

        # weights for sigma points matrix
        self.weights = np.zeros(2*n_aug + 1)
        self.weights[0] = self.lamba/(self.lamba + n_aug)
        for i in range(1, len(self.weights)):
            self.weights[i] = 0.5/(n_aug + self.lamba)

        # sigma points matrix
        self.Xsig_pts = np.empty([self.n_x, 2*n_aug + 1])

    @property
    def sensors(self):
        return self._sensors

    @sensors.setter
    def sensors(self, sensor_list):
        self._sensors = sensor_list

    def process_measurement(self, measurement):
        for s in self._sensors:
            s.update_measurement(measurement)

    def generate_augmented_sigma_pts(self):
        # dimension of augmented state vector
        n_aug = self.n_x + 2 # 2D process noise

        # augmented state vector
        x_aug = np.empty(n_aug)
        for i in range(self.n_x):
            x_aug[i] = self.x[i]
        x_aug[self.n_x] = 0
        x_aug[self.n_x + 1] = 0

        # augmented covariance
        P_aug = np.zeros([n_aug, n_aug])
        P_aug[:-2, :-2] = self.P
        P_aug[self.n_x, self.n_x] = self.std_a * self.std_a
        P_aug[self.n_x + 1, self.n_x + 1] = self.std_yawdd * self.std_yawdd

        # TODO: this is a hack to ensure Positive Definite
        P_aug = 0.5 * (np.matmul(P_aug, np.transpose(P_aug)))
        # Cholesky decomposition
        L = np.linalg.cholesky(P_aug)

        # generate augmented sigma points
        Xsig_aug = np.zeros([n_aug, 2*n_aug + 1])
        Xsig_aug[:,0] = x_aug

        for i in range(n_aug):
            Xsig_aug[:,i+1] = x_aug + np.sqrt(self.lamba+n_aug) * L[:,i]
            Xsig_aug[:,i+1+n_aug] = x_aug - np.sqrt(self.lamba+n_aug) * L[:,i]

        return Xsig_aug

    def predict_sigma_pts(self, Xsig_aug):
        # dimension of augmented state vector
        n_aug = self.n_x + 2 # 2D process noise

        # create matrix with predicted sigma points as columns
        Xsig_pred = np.zeros([self.n_x, 2*n_aug + 1])

        # predict sigma points
        for i in range(2*n_aug + 1):
            # extract values and initialization for better readability
            px = Xsig_aug[0,i]
            py = Xsig_aug[1,i]
            v = Xsig_aug[2,i]
            yaw = Xsig_aug[3,i]
            yawd = Xsig_aug[4,i]
            nu_a = Xsig_aug[5,i]
            nu_yawdd = Xsig_aug[6,i]

            if np.abs(yawd) > 1e-3:
                px_pred = px + v/yawd * ( np.sin(yaw + yawd*self.dt) - np.sin(yaw) )
                py_pred = py + v/yawd * ( np.cos(yaw) - np.cos(yaw + yawd * self.dt) )
            else:
                px_pred = px + v*self.dt*np.cos(yaw)
                py_pred = py + v*self.dt*np.sin(yaw)

            # costant v, yawd
            v_pred = v
            yaw_pred = yaw + yawd*self.dt
            yawd_pred = yawd

            # add noise
            px_pred = px_pred + 0.5*nu_a*self.dt*self.dt*np.cos(yaw)
            py_pred = py_pred + 0.5*nu_a*self.dt*self.dt*np.sin(yaw)
            v_pred = v_pred + nu_a*self.dt

            yaw_pred = yaw_pred + 0.5*nu_yawdd*self.dt*self.dt
            yawd_pred = yawd_pred + nu_yawdd*self.dt

            Xsig_pred[0,i] = px_pred
            Xsig_pred[1,i] = py_pred
            Xsig_pred[2,i] = v_pred
            Xsig_pred[3,i] = yaw_pred
            Xsig_pred[4,i] = yawd_pred

            # print("iter: {} ->".format(i))
            # print(Xsig_pred)

        self.Xsig_pts = Xsig_pred

        return self.Xsig_pts

    def predict_mean_cov(self, Xsig_pred):
        # dimension of augmented state vector
        n_aug = self.n_x + 2 # 2D process noise

        # create vector for predicted state
        x_out = np.zeros(self.n_x)

        P_out = np.zeros([self.n_x, self.n_x])

        # predicted state covariance matrix
        # iterate over sigma points
        for i in range(2*n_aug + 1):
            # state difference
            x_diff = Xsig_pred[:,i] - x_out

            # angle normalization
            while x_diff[3] > np.pi:
                x_diff[3] = x_diff[3] - 2.*np.pi
            while x_diff[3] < -np.pi:
                x_diff[3] = x_diff[3] + 2.*np.pi

            P_out = P_out + self.weights[i]*np.outer(x_diff, np.transpose(x_diff))

        # update state
        self.x = x_out
        self.P = P_out

    def predict(self):
        Xsig_aug = self.generate_augmented_sigma_pts()
        Xsig_pred = self.predict_sigma_pts(Xsig_aug)
        self.predict_mean_cov(Xsig_pred)

    # currently use I80 data, should use Sensor abstract class and treat I80 data as a type of sensor data
    def update(self):
        for s in self._sensors:
            s.update_ukf()

    def step_filter(self):
        self.predict()
        self.update()
