sensor_types = {'HW-cam': 0, 'GPS': 1, 'IMU': 2, 'Lidar': 3}


class Sensor(object):
    """
    virtual class: Sensor interface to be used with a UKF
        a Sensor object requires to reference a UKF object for sensor fusion
    """

    def __init__(self, name='', active=False, noise=None, sensor_type=None, ukf=None):
        self.name = name
        self.is_active = active
        self.R = noise
        self.sensor_type = sensor_type
        self.ukf = ukf
        self.data_dim = None
        self.initialized = False
        self.has_new_data = False
        self.measurement = None

    def enable(self):
        self.is_active = True

    def disable(self):
        self.is_active = False

    def update_measurement(self, meas):
        pass

    def update(self, ukf):
        pass
