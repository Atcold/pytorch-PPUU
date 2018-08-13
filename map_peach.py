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
X_OFFSET = 0  # horizontal offset (camera 2 leftmost view)
MAX_SPEED = 130


class PeachCar(RealCar):
    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W
    X_OFFSET = X_OFFSET
    max_b = 0.05  # set a looser max turning limitation

    @property
    def current_lane(self):
        # 1: left-most, 5: right-most, 6: auxiliary lane, 7: on-ramp, 8: off-ramp
        return 0


class Peachtree(RealTraffic):
    # Environment's car class
    EnvCar = PeachCar

    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W
    X_OFFSET = X_OFFSET
    DUMP_NAME = 'data_lanker_v0'

    def __init__(self, **kwargs):
        kwargs['nb_lanes'] = 1
        kwargs['delta_t'] = 1/10
        super(RealTraffic, self).__init__(**kwargs)

        w = (640, 624, 472, 688, 456, 472, 752, 280)
        self.screen_size = (sum(w[-8:]) - 270, 20 * self.LANE_W)
        self.photos = (
            pygame.image.load('Peachtree/cam8.png'),
            pygame.image.load('Peachtree/cam7.png'),
            pygame.image.load('Peachtree/cam6.png'),
            pygame.image.load('Peachtree/cam5.png'),
            pygame.image.load('Peachtree/cam4.png'),
            pygame.image.load('Peachtree/cam3.png'),
            pygame.image.load('Peachtree/cam2.png'),
            pygame.image.load('Peachtree/cam1.png'),
        )
        self.photos_rect = (
            self.photos[0].get_rect().move([0, 25]),
            self.photos[1].get_rect().move([w[-1] - 25, 25]),
            self.photos[2].get_rect().move([sum(w[-2:]) - 30, 25]),
            self.photos[3].get_rect().move([sum(w[-3:]) - 35, 25]),
            self.photos[4].get_rect().move([sum(w[-4:]) - 120, 25]),
            self.photos[5].get_rect().move([sum(w[-5:]) - 220, 25]),
            self.photos[5].get_rect().move([sum(w[-6:]) - 230, 25]),
            self.photos[5].get_rect().move([sum(w[-7:]) - 270, 25]),
        )
        if self.display:  # if display is required
            self.screen = pygame.display.set_mode(self.screen_size)  # set screen size
        # self.delta_t = 1 / 10  # simulation timing interval
        self._time_slots = (
            './traffic-data/xy-trajectories/peach/trajectories-0400pm-0415pm',
            './traffic-data/xy-trajectories/peach/trajectories-1245pm-0100pm',
        )
        self._t_slot = None
        self._black_list = {
            self._time_slots[0]:
                set(),
            self._time_slots[1]:
                set(),
        }
        self.df = None
        self.vehicles_history = None
        self.lane_occupancy = None
        # self._lane_surfaces = dict()
        # self.nb_lanes = 1
        self.smoothing_window = 15
        self.offset = -180

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
        df = df[valid_x]

        # Invert coordinates (IDK WTF is going on with these trajectories)
        max_x = df['Local Y'].max()
        max_y = df['Local X'].max()
        df['Local Y'] = max_x + 30 - df['Local Y']
        df['Local X'] = max_y - df['Local X']

        # Restrict data frame to valid x coordinates
        return df

    def _draw_lanes(self, surface, mode='human', offset=0):

        slope = 0.07

        lanes = self.lanes  # lanes

        if mode == 'human':

            # # load lanes, if not already done so
            # if mode not in self._lane_surfaces:
            #     self._lane_surfaces[mode] = pygame.image.load('Lankershim/lanes_human.png')
            #
            # surface.blit(self._lane_surfaces[mode], (0, 0))
            pass

        if mode == 'machine':

            # load lanes
            lanes_surface = pygame.image.load('Lankershim/lanes_machine.png')
            surface.blit(lanes_surface, (offset, offset))

            # save for later
            self._lane_surfaces[mode] = surface.copy()
            # pygame.image.save(surface, "us101-machine.png")
