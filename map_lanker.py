from random import choice, randrange

from custom_graphics import draw_dashed_line
from traffic_gym_v1 import RealTraffic, RealCar, colours
from traffic_gym import StatefulEnv
import pygame
import pandas as pd
import numpy as np
import pdb, random
import bisect
import pdb, pickle, os
from tqdm import tqdm

# Conversion LANE_W from real world to pixels
# A US highway lane width is 3.7 metres, here 50 pixels
LANE_W = 24  # pixels / 3.7 m, lane width
SCALE = LANE_W / 3.7  # pixels per metre
FOOT = 0.3048  # metres per foot
X_OFFSET = -35  # horizontal offset (camera 2 leftmost view)
MAX_SPEED = 130


class LankerCar(RealCar):
    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W
    X_OFFSET = X_OFFSET
    max_b = 0.05  # set a looser max turning limitation

    @property
    def current_lane(self):
        # 1: left-most, 5: right-most, 6: auxiliary lane, 7: on-ramp, 8: off-ramp
        return 0


class Lankershim(RealTraffic):
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
        super(RealTraffic, self).__init__(**kwargs)

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
            './traffic-data/xy-trajectories/lanker/trajectories-0830am-0845am',
            './traffic-data/xy-trajectories/lanker/trajectories-0845am-0900am',
        )
        self._t_slot = None
        self._black_list = {
            self._time_slots[0]:
                {128, 65, 995, 1124, 377, 810, 1003, 172, 335, 591,  # off track (OT)
                 560, 1173, 1399, 1437, 153, 890, 1308, 1405, 413, 639},  # OT
            self._time_slots[1]:
                {1539, 772, 517, 267, 396, 1164, 1421, 1549, 530, 664, 1570, 1059, 804, 169, 812, 1453, 48, 53,  # OT
                 1469, 1600, 1472, 1474, 451, 580, 1478, 584, 212, 1492, 1114, 228, 233, 625, 1394, 1268, 1023},  # OT
        }
        self.df = None
        self.vehicles_history = None
        self.lane_occupancy = None
        # self._lane_surfaces = dict()
        # self.nb_lanes = 1
        self.smoothing_window = 15
        self.offset = 195

    @staticmethod
    def _get_data_frame(file_name, x_max, x_offset):
        print(f'Loading trajectories from {file_name}')
        df = pd.read_table(file_name, sep='\s+', header=None, names=(
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

        slope = 0.07

        lanes = self.lanes  # lanes

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
            # pygame.image.save(surface, "us101-machine.png")
