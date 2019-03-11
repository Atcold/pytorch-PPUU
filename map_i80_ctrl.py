from map_i80 import I80, I80Car
from traffic_gym_v2 import PatchedCar


class ControlledI80Car(I80Car):

    # Import get_lane_set from PatchedCar
    get_lane_set = PatchedCar.get_lane_set

    def __init__(self, df, y_offset, look_ahead, screen_w, font=None, kernel=0, dt=1/10):
        super().__init__(df, y_offset, look_ahead, screen_w, font, kernel, dt)
        self.is_controlled = False
        self.buffer_size = 0
        self.lanes = None
        self.arrived_to_dst = False  # arrived to destination
        self.frames = list()

    @property
    def current_lane(self):
        # If following the I-80 trajectories
        if not self.is_controlled or len(self._states_image) < self.buffer_size:
            return super().current_lane

        # Otherwise fetch x location
        x = self._position[0]
        if x > self.screen_w - 1.75 * self.look_ahead:
            self.off_screen = True
            self.arrived_to_dst = True

        # Fetch the y location
        y = self._position[1]

        # If way too up
        if y < self.lanes[0]['min']:
            self.off_screen = True
            self.arrived_to_dst = False
            return 0

        # Maybe within a sensible range?
        for lane_idx, lane in enumerate(self.lanes):
            if lane['min'] <= y <= lane['max']:
                return lane_idx

        # Or maybe on the ramp
        bottom = self.lanes[-1]['max']
        if y <= bottom + 53 - x * 0.035:
            return 6

        # Actually, way too low
        self.off_screen = True
        self.arrived_to_dst = False
        return 6

    @property
    def is_autonomous(self):
        return self.is_controlled and len(self._states_image) > self.buffer_size


class ControlledI80(I80):

    # Environment's car class
    EnvCar = ControlledI80Car

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        observation = None
        while observation is None:
            observation, reward, done, info = self.step()
        return observation
