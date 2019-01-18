from random import choice, randrange

from custom_graphics import draw_dashed_line
from map_i80 import I80, I80Car, colours
from traffic_gym import Simulator
import pygame
import pandas as pd
import numpy as np
import pdb, random
import bisect
import pdb, pickle, os

# Conversion LANE_W from real world to pixels
# A US highway lane width is 3.7 metres, here 50 pixels
LANE_W = 24  # pixels / 3.7 m, lane width
SCALE = LANE_W / 3.7  # pixels per metre
FOOT = 0.3048  # metres per foot
X_OFFSET = 615  # horizontal offset (camera 2 leftmost view)
MAX_SPEED = 130


class US101Car(I80Car):
    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W
    X_OFFSET = X_OFFSET

    @property
    def current_lane(self):
        # 1: left-most, 5: right-most, 6: auxiliary lane, 7: on-ramp, 8: off-ramp
        return min(self._lane_list[self._frame], 6) - 1


class US101(I80):
    # Environment's car class
    EnvCar = US101Car

    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W
    X_OFFSET = X_OFFSET
    DUMP_NAME = 'data_us101_v0'

    def __init__(self, **kwargs):
        kwargs['nb_lanes'] = 5
        kwargs['delta_t'] = 1/10
        super().__init__(**kwargs)

        self.screen_size = (125 * self.LANE_W, self.nb_lanes * self.LANE_W + 4 * self.LANE_W)
        # self.photos = (
        #     pygame.image.load('US-101/cam7.png'),
        #     pygame.image.load('US-101/cam6.png'),
        #     pygame.image.load('US-101/cam5.png'),
        #     pygame.image.load('US-101/cam4.png'),
        #     pygame.image.load('US-101/cam3.png'),
        # )
        # self.photos_rect = (
        #     self.photos[0].get_rect().move([0, 33]),
        #     self.photos[1].get_rect().move([552, 33 + 2]),
        #     self.photos[2].get_rect().move([552 + 472, 33 + 2]),
        #     self.photos[3].get_rect().move([552 + 472 + 388, 33 + 3]),
        #     self.photos[4].get_rect().move([552 + 472 + 388 + 532, 33 + 3]),
        # )
        if self.display:  # if display is required
            self.screen = pygame.display.set_mode(self.screen_size)  # set screen size
        # self.delta_t = 1 / 10  # simulation timing interval
        self._time_slots = (
            'us101/trajectories-0750am-0805am',
            'us101/trajectories-0805am-0820am',
            'us101/trajectories-0820am-0835am',
        )
        self._t_slot = None
        self._black_list = {
            self._time_slots[0]:
                {2691, 2809, 2820, 2871, 2851, 2873},
            self._time_slots[1]:
                {649, 806, 1690, 1725, 1734, 1773, 1949, 1877},
            self._time_slots[2]:
                {183, 329, 791, 804, 1187, 1183, 1107, 1247, 1202, 1371, 1346, 1435, 1390, 1912},
        }
        self.df = None
        self.vehicles_history = None
        self.lane_occupancy = None
        self.nb_lanes = 6
        self.smoothing_window = 15

    def _draw_lanes(self, surface, mode='human', offset=0):

        slope = 0.07

        lanes = self.lanes  # lanes

        if mode == 'human':
            s = surface  # screen
            draw_line = pygame.draw.line  # shortcut
            w = colours['w']  # colour white
            g = (128, 128, 128)
            sw = self.screen_size[0]  # screen width

            for lane in lanes:
                draw_line(s, g, (0, lane['min']), (sw, lane['min']), 1)
                # draw_dashed_line(s, colours['r'], (0, lane['mid']), (sw, lane['mid']))  # red centres

            draw_line(s, w, (0, lanes[0]['min']), (sw, lanes[0]['min']), 3)
            bottom = lanes[-1]['max']
            draw_line(s, w, (0, bottom), (28 * LANE_W, bottom), 3)
            draw_line(s, g, (28 * LANE_W, bottom), (86 * LANE_W, bottom), 1)
            draw_line(s, w, (86 * LANE_W, bottom), (sw, bottom), 3)
            draw_line(s, w, (22 * LANE_W, bottom + LANE_W), (90 * LANE_W, bottom + LANE_W), 3)

            # Tilted lanes and lines
            x0, x1, y = 0 * LANE_W, 20 * LANE_W, bottom + 35
            draw_line(s, w, (x0, y), (x1, y - slope * (x1 - x0)), 3)
            x0, x1, y = 15 * LANE_W, 22 * LANE_W, bottom + 35
            draw_line(s, w, (x0, y), (x1, y - slope * (x1 - x0)), 3)
            x0, x1, y = 92 * LANE_W, 112 * LANE_W, bottom
            draw_line(s, w, (x0, y), (x1, y + slope * (x1 - x0)), 3)
            x0, x1, y = 90 * LANE_W, 97 * LANE_W, bottom + LANE_W
            draw_line(s, w, (x0, y), (x1, y + slope * (x1 - x0)), 3)

            look_ahead = MAX_SPEED * 1000 / 3600 * self.SCALE
            o = self.offset
            draw_line(s, (255, 255, 0), (look_ahead, o), (look_ahead, 9.4 * LANE_W))
            draw_line(s, (255, 255, 0), (sw - 1.75 * look_ahead, o), (sw - 1.75 * look_ahead, bottom))
            draw_line(s, (255, 255, 0), (sw - 0.75 * look_ahead, o), (sw - 0.75 * look_ahead, bottom), 5)

        if mode == 'machine':
            s = surface  # screen
            draw_line = pygame.draw.line  # shortcut
            w = colours['r']  # colour white
            sw = self.screen_size[0]  # screen width
            m = offset

            for lane in lanes:
                draw_line(s, w, (0, lane['min'] + m), (sw + 2 * m, lane['min'] + m), 1)

            draw_line(s, w, (0, lanes[-1]['max'] + m), (sw + 2 * m, lanes[-1]['max'] + m), 1)
            bottom = lanes[-1]['max'] + m
            draw_line(s, w, (0, bottom), (28 * LANE_W, bottom), 1)
            draw_line(s, w, (28 * LANE_W, bottom), (86 * LANE_W, bottom), 1)
            draw_line(s, w, (86 * LANE_W, bottom), (sw, bottom), 1)
            draw_line(s, w, (22 * LANE_W, bottom + LANE_W), (90 * LANE_W, bottom + LANE_W), 1)

            # Tilted lanes and lines
            x0, x1, y = 0 * LANE_W, 20 * LANE_W, bottom + 35
            draw_line(s, w, (x0, y), (x1, y - slope * (x1 - x0)), 1)
            x0, x1, y = 15 * LANE_W, 22 * LANE_W, bottom + 35
            draw_line(s, w, (x0, y), (x1, y - slope * (x1 - x0)), 1)
            x0, x1, y = 92 * LANE_W, 112 * LANE_W, bottom
            draw_line(s, w, (x0, y), (x1, y + slope * (x1 - x0)), 1)
            x0, x1, y = 90 * LANE_W, 97 * LANE_W, bottom + LANE_W
            draw_line(s, w, (x0, y), (x1, y + slope * (x1 - x0)), 1)

            self._lane_surfaces[mode] = surface.copy()
            # pygame.image.save(surface, "us101-machine.png")
