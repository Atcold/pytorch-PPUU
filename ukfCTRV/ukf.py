import numpy as np


class CTRV(object):

    def __init__(self):
        self.px = 0.
        self.py = 0.


class ukf(object):

    def __init__(self):
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

    def process_measurement(self, measurement):
        pass

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
        print(P_aug)

        # Cholesky decomposition
        L = np.linalg.cholesky(P_aug)
        print(L)

        # generate augmented sigma points
        Xsig_aug = np.zeros([n_aug, 2*n_aug + 1])
        Xsig_aug[:,0] = x_aug

        for i in range(n_aug):
            Xsig_aug[:,i+1] = x_aug + np.sqrt(self.lamba+n_aug) * L[:,i]
            Xsig_aug[:,i+1+n_aug] = x_aug - np.sqrt(self.lamba+n_aug) * L[:,i]
        print(Xsig_aug)

        return Xsig_aug

car = ukf()
car.generate_augmented_sigma_pts()


