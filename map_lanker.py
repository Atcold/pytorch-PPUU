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
X_OFFSET = -35  # horizontal offset (camera 2 leftmost view)
MAX_SPEED = 130


class LankerCar(I80Car):
    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W
    X_OFFSET = X_OFFSET
    max_b = 0.05  # set a looser max turning limitation

    @property
    def current_lane(self):
        # 1: left-most, 5: right-most, 6: auxiliary lane, 7: on-ramp, 8: off-ramp
        return 0


class Lankershim(I80):
    # Environment's car class
    EnvCar = LankerCar

    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W
    X_OFFSET = X_OFFSET
    DUMP_NAME = 'data_lanker_v0'

    def __init__(self, **kwargs):
        kwargs['nb_lanes'] = 1
        kwargs['delta_t'] = 1/10
        super().__init__(**kwargs)

        self.screen_size = (560 + 760 + 648 + 912 + 328, 20 * self.LANE_W)
        # self.photos = (
        #     pygame.image.load('Lankershim/cam1.png'),
        #     pygame.image.load('Lankershim/cam2.png'),
        #     pygame.image.load('Lankershim/cam3.png'),
        #     pygame.image.load('Lankershim/cam4.png'),
        #     pygame.image.load('Lankershim/cam5.png'),
        # )
        # self.photos_rect = (
        #     self.photos[0].get_rect().move([0, 20]),
        #     self.photos[1].get_rect().move([560, 20]),
        #     self.photos[2].get_rect().move([560 + 760, 20]),
        #     self.photos[3].get_rect().move([560 + 760 + 648, 20]),
        #     self.photos[4].get_rect().move([560 + 760 + 648 + 912, 20]),
        # )
        if self.display:  # if display is required
            self.screen = pygame.display.set_mode(self.screen_size)  # set screen size
        # self.delta_t = 1 / 10  # simulation timing interval
        self._time_slots = (
            'lanker/trajectories-0830am-0845am',
            'lanker/trajectories-0845am-0900am',
        )
        self._t_slot = None
        self._black_list = {
            self._time_slots[0]:
                {128, 65, 995, 1124, 377, 810, 1003, 172, 335, 591,  # off track (OT)
                 560, 1173, 1399, 1437, 153, 890, 1308, 1405, 413, 639,  # OT
                 66, 112, 111, 94, 115, 122, 130, 170, 149, 152, 160, 210, 292, 261, 291, 339,  # crash
                 300, 312, 306, 320, 391, 415, 434, 436, 472, 345, 432, 468, 397, 329, 528, 567,  # crash
                 549, 468, 530, 585, 624, 737, 711, 716, 690, 753, 716, 762, 818, 904, 930, 887,  # crash
                 964, 906, 931, 1005, 982, 989, 1000, 1433, 1037, 1189, 1155, 1221, 1260, 1258,  # crash
                 1249, 1277, 1285, 1386, 1372, 1366, 1007, 1001},  # crash
            self._time_slots[1]:
                {1539, 772, 517, 267, 396, 1164, 1421, 1549, 530, 664, 1570, 1059, 804, 169, 812, 1453, 48, 53,  # OT
                 1469, 1600, 1472, 1474, 451, 580, 1478, 584, 212, 1492, 1114, 228, 233, 625, 1394, 1268, 1023,  # OT
                 58, 36, 129, 131, 74, 163, 122, 160, 296, 321, 330, 369, 395, 358, 322, 274, 481, 492,  # crash
                 443, 490, 524, 437, 545, 600, 487, 730, 740, 628, 810, 753, 844, 716, 903, 672, 915, 936,  # crash
                 809, 872, 967, 1075, 1069, 1109, 1098, 1075, 982, 986, 1069, 1109, 1180, 1155, 1103, 1232,  # crash
                 1238, 1260, 1132, 1308, 1353, 1306, 1392, 1409, 1301, 1456, 1422, 1475, 1542, 1552, 1524,  # crash
                 348, 521, 824, 911, 985, 1178}
        }
        self.df = None
        self.vehicles_history = None
        self.lane_occupancy = None
        # self._lane_surfaces = dict()
        # self.nb_lanes = 1
        self.smoothing_window = 15
        self.offset = 195

    @staticmethod
    def _get_data_frame(time_slot, x_max, x_offset):
        # TODO: need caching! See I-80
        file_name = f'traffic-data/xy-trajectories/{time_slot}.txt'
        print(f'Loading trajectories from {file_name}')
        df = pd.read_csv(file_name, sep=r'\s+', header=None, names=(
            'Vehicle ID',
            'Frame ID',
            'Total Frames',
            'Global Time',
            'Local X',
            'Local Y',
            'Global X',
            'Global Y',
            'Vehicle Length',
            'Vehicle Width',
            'Vehicle Class',
            'Vehicle Velocity',
            'Vehicle Acceleration',
            'Lane Identification',
            'Origin Zone',
            'Destination Zone',
            'Intersection',
            'Section',
            'Direction',
            'Movement',
            'Preceding Vehicle',
            'Following Vehicle',
            'Spacing',
            'Headway'
        ))

        # Get valid x coordinate rows
        valid_x = (df['Local Y'] * FOOT * SCALE - x_offset).between(0, x_max).values

        # Restrict data frame to valid x coordinates
        return df[valid_x]

    def _draw_lanes(self, surface, mode='human', offset=0):

        if mode == 'human':

            # load lanes, if not already done so
            if mode not in self._lane_surfaces:
                self._lane_surfaces[mode] = pygame.image.load('Lankershim/lanes_human.png')

            surface.blit(self._lane_surfaces[mode], (0, 0))

        if mode == 'machine':

            # load lanes
            lanes_surface = pygame.image.load('Lankershim/lanes_machine.png')
            surface.blit(lanes_surface, (offset, offset))

            # save for later
            self._lane_surfaces[mode] = surface.copy()
