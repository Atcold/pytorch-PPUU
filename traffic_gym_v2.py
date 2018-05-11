from traffic_gym import StatefulEnv, Car
from traffic_gym_v1 import RealTraffic
import pygame

# Conversion LANE_W from real world to pixels
# A US highway lane width is 3.7 metres, here 50 pixels
LANE_W = 24  # pixels / 3.7 m, lane width
SCALE = LANE_W / 3.7  # pixels per metre


class PatchedCar(Car):
    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_lane_set(self, lanes):
        # Bottom end of normal lanes
        bottom = lanes[-1]['max']

        # No merging
        if self._position[1] < bottom:
            return super().get_lane_set(lanes)

        # Done merging
        if self._position[0] > 60 * LANE_W:
            self._target_lane = lanes[-1]['mid']
            return {5}

        # We're on ramp!
        self._target_lane = bottom + 42 - self._position[0] * 0.035

        if self._position[0] < 18 * LANE_W:
            return {6}
        else:
            return {5, 6}


class MergingMap(StatefulEnv):
    # Environment's car class
    EnvCar = PatchedCar

    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W

    # Import map from RealTraffic
    _draw_lanes = RealTraffic._draw_lanes

    def __init__(self, **kwargs):
        kwargs['nb_lanes'] = 6
        kwargs['delta_t'] = 1/10
        super().__init__(**kwargs)
        self.nb_lanes = 7
        self.screen_size = (85 * self.LANE_W, self.nb_lanes * self.LANE_W + 5 * self.LANE_W)
        if self.display:  # if display is required
            self.screen = pygame.display.set_mode(self.screen_size)  # set screen size
