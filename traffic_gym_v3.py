from traffic_gym_v1 import RealTraffic, RealCar
from traffic_gym_v2 import PatchedCar


class ControlledCar(RealCar):

    # Import get_lane_set from PatchedCar
    get_lane_set = PatchedCar.get_lane_set

    def __init__(self, df, y_offset, look_ahead, screen_w, font=None):
        super().__init__(df, y_offset, look_ahead, screen_w, font)
        self.is_controlled = False
        self.buffer_size = 0
        self.lanes = None
        self.screen_w = None

    @property
    def current_lane(self):
        # If following the I-80 trajectories
        if not self.is_controlled or len(self._states_image) < self.buffer_size:
            return super().current_lane

        # Otherwise fetch x location
        x = self._position[0]
        if x > self.screen_w: self.off_screen = True

        # Fetch the y location
        y = self._position[1]

        # If way too up
        if y < self.lanes[0]['min']:
            self.off_screen = True
            return 0

        # Maybe within a sensible range?
        for lane_idx, lane in enumerate(self.lanes):
            if lane['min'] <= y <= lane['max']:
                return lane_idx

        # Or maybe on the ramp
        bottom = self.lanes[-1]['max']
        if y <= bottom + 53 - self._position[0] * 0.035:
            return 6

        # Actually, way too low
        self.off_screen = True
        return 6


class ControlledI80(RealTraffic):

    # Environment's car class
    EnvCar = ControlledCar

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
