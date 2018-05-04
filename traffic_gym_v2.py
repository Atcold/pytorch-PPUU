from traffic_gym import StatefulEnv, Car
from traffic_gym_v1 import RealTraffic
import pygame

# Conversion LANE_W from real world to pixels
# A US highway lane width is 3.7 metres, here 50 pixels
LANE_W = 24  # pixels / 3.7 m, lane width
SCALE = LANE_W / 3.7  # pixels per metre


class MergingMap(StatefulEnv):
    # Environment's car class
    EnvCar = Car

    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W

    # Import map from RealTraffic
    _draw_lanes = RealTraffic._draw_lanes

    def __init__(self, **kwargs):
        kwargs['nb_lanes'] = 6
        kwargs['delta_t'] = 1/10
        super().__init__(**kwargs)
        self.screen_size = (85 * self.LANE_W, self.nb_lanes * self.LANE_W + 5 * self.LANE_W)
        if self.display:  # if display is required
            self.screen = pygame.display.set_mode(self.screen_size)  # set screen size
