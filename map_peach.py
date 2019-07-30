from random import choice, randrange

from custom_graphics import draw_dashed_line
from map_lanker import LankerCar
from map_i80 import I80, colours
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
X_OFFSET = 0  # horizontal offset (camera 2 leftmost view)
MAX_SPEED = 130


class PeachCar(LankerCar):
    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W
    X_OFFSET = X_OFFSET
    max_b = 0.05  # set a looser max turning limitation


class Peachtree(I80):
    # Environment's car class
    EnvCar = PeachCar

    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W
    X_OFFSET = X_OFFSET
    DUMP_NAME = 'data_peach_v0'

    def __init__(self, **kwargs):
        kwargs['nb_lanes'] = 1
        kwargs['delta_t'] = 1/10
        super().__init__(**kwargs)

        w = (640, 624, 472, 688, 456, 472, 752, 280)
        self.screen_size = (sum(w[-8:]) - 270, 315)
        # self.photos = (
        #     pygame.image.load('Peachtree/cam8.png'),
        #     pygame.image.load('Peachtree/cam7.png'),
        #     pygame.image.load('Peachtree/cam6.png'),
        #     pygame.image.load('Peachtree/cam5.png'),
        #     pygame.image.load('Peachtree/cam4.png'),
        #     pygame.image.load('Peachtree/cam3.png'),
        #     pygame.image.load('Peachtree/cam2.png'),
        #     pygame.image.load('Peachtree/cam1.png'),
        # )
        # self.photos_rect = (
        #     self.photos[0].get_rect().move([0, 25]),
        #     self.photos[1].get_rect().move([w[-1] - 25, 25]),
        #     self.photos[2].get_rect().move([sum(w[-2:]) - 30, 25]),
        #     self.photos[3].get_rect().move([sum(w[-3:]) - 35, 25]),
        #     self.photos[4].get_rect().move([sum(w[-4:]) - 120, 25]),
        #     self.photos[5].get_rect().move([sum(w[-5:]) - 220, 25]),
        #     self.photos[5].get_rect().move([sum(w[-6:]) - 230, 25]),
        #     self.photos[5].get_rect().move([sum(w[-7:]) - 270, 25]),
        # )
        if self.display:  # if display is required
            self.screen = pygame.display.set_mode(self.screen_size)  # set screen size
        # self.delta_t = 1 / 10  # simulation timing interval
        self._time_slots = (
            'peach/trajectories-0400pm-0415pm',
            'peach/trajectories-1245pm-0100pm',
        )
        self._t_slot = None
        self._black_list = {
            self._time_slots[0]:
                {256, 137, 11, 1293, 399, 1551, 794, 556, 942, 562, 307, 1077, 694, 188, 63, 705, 451, 579, 1098, 605,
                 606, 95, 225, 611, 997, 107, 1643, 366, 624, 245, 255, 738, 1755,  # off track
                 114, 15, 70, 142, 28, 94, 1672, 93, 194, 75, 59, 249, 75, 239, 253, 240, 254, 269, 266, 275, 336, 1709,
                 290, 247, 262, 1689, 352, 380, 415, 468, 405, 449, 419, 475, 396, 559, 597, 703, 404, 642, 682, 593,
                 567, 703, 650, 815, 760, 763, 945, 951, 973, 975, 1005, 961, 978, 972, 1103, 1144, 984, 970, 1122, 978,
                 1179, 1090, 1147, 1145, 1201, 1314, 1134, 1201, 1200, 1431, 1397, 1507, 1640, 1552, 1392, 1530, 1561,
                 1564},
            self._time_slots[1]:
                {391, 1037, 399, 404, 1459, 948, 1206, 440, 314, 1339, 829, 577, 962, 67, 219, 861, 863, 991, 358, 998,
                 246, 1022, 127,  # off track
                 197, 131, 211, 228, 125, 218, 112, 299, 217, 406, 402, 297, 376, 409, 444, 569, 436, 426, 620, 706,
                 701, 787, 783, 723, 1498, 734, 715, 757, 706, 798, 805, 787, 842, 847, 783, 901, 850, 994, 882, 1092,
                 1055, 966, 1008, 1092, 1079, 1026, 1142, 1273, 1164, 1183, 1320, 1324, 1192, 1129, 1320, 1372, 1326,
                 1406, 1372, 1358, 1298, 1336, 1480},
        }
        self.df = None
        self.vehicles_history = None
        self.lane_occupancy = None
        # self._lane_surfaces = dict()
        # self.nb_lanes = 1
        self.smoothing_window = 15
        self.offset = None  # data is fucked up here, fixing it in the custom reset method

    def reset(self, frame=None, time_slot=None):
        super().reset(frame, time_slot)
        self.offset = -180 if time_slot == 0 else -15

    def _get_data_frame(self, time_slot, x_max, x_offset):
        # TODO: should use Lanker, not re-implement this!
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
        df = df[valid_x]

        # Invert coordinates (IDK WTF is going on with these trajectories)
        max_x = df['Local Y'].max()
        max_y = df['Local X'].max()
        extra_offset = 30 if time_slot == 0 else 17
        df['Local Y'] = max_x + extra_offset - df['Local Y']
        df['Local X'] = max_y - df['Local X']

        # Dropping cars with lifespan shorter than 5 second
        baby_cars = set(df[df['Total Frames'] < 50]['Vehicle ID'])
        print(f'Removing {len(baby_cars)} baby vehicles from the database')
        self._black_list[time_slot] |= baby_cars

        # Restrict data frame to valid x coordinates
        return df

    def _draw_lanes(self, surface, mode='human', offset=0):

        if mode == 'human':

            # load lanes, if not already done so
            if mode not in self._lane_surfaces:
                self._lane_surfaces[mode] = pygame.image.load('Peachtree/lanes_human.png')

            surface.blit(self._lane_surfaces[mode], (0, 0))

        if mode == 'machine':

            # load lanes
            lanes_surface = pygame.image.load('Peachtree/lanes_machine.png')
            surface.blit(lanes_surface, (offset, offset))

            # save for later
            self._lane_surfaces[mode] = surface.copy()
