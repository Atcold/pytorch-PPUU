from custom_graphics import draw_dashed_line
from traffic_gym import StatefulEnv, Car, colours
import pygame
import pandas as pd
import numpy as np

# Conversion LANE_W from real world to pixels
# A US highway lane width is 3.7 metres, here 50 pixels
LANE_W = 24  # pixels / 3.7 m, lane width
SCALE = LANE_W / 3.7  # pixels per metre
FOOT = 0.3048  # metres per foot
X_OFFSET = 370  # horizontal offset (camera 2 leftmost view)


class RealCar(Car):
    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W

    def __init__(self, df, y_offset):
        self._df = df
        self.id = df.at[df.index[0], 'Vehicle ID']  # extract scalar <'Vehicle ID'> <at> <index[0]>
        self._length = df.at[df.index[0], 'Vehicle Length'] * FOOT * SCALE
        self._width = df.at[df.index[0], 'Vehicle Width'] * FOOT * SCALE
        self._offset = np.array((X_OFFSET + self._length, -y_offset))  # pixels, by hand
        # X and Y are swapped in the I-80 data set...
        self._position = df.loc[df.index[0], ['Local Y', 'Local X']].values * FOOT * SCALE - self._offset
        self._frame = 0
        self._direction = np.array((1., 0.))
        self._colour = colours['c']
        self._braked = False
        self.off_screen = False

    def step(self, action):
        self._frame += 1
        position = self._position
        df = self._df
        self._position = df.loc[df.index[self._frame], ['Local Y', 'Local X']].values * FOOT * SCALE - self._offset
        new_direction = self._position - position
        self._direction = new_direction if np.linalg.norm(new_direction) > 1 else self._direction
        self._direction /= np.linalg.norm(self._direction)
        self.off_screen = self._frame == len(df) - 1

    def policy(self, observation):
        return None


class RealTraffic(StatefulEnv):
    # Environment's car class
    EnvCar = RealCar

    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W

    def __init__(self, **kwargs):
        kwargs['nb_lanes'] = 6
        kwargs['delta_t'] = 1/10
        super().__init__(**kwargs)

        self.screen_size = (67 * self.LANE_W, self.nb_lanes * self.LANE_W + 5 * self.LANE_W)
        # self.photos = (
        #     pygame.image.load('vlcsnap-2018-02-22-14h40m23s503.png'),
        #     pygame.image.load('vlcsnap-2018-02-23-10h55m01s517.png'),
        #     pygame.image.load('vlcsnap-2018-03-08-16h22m49s299.png')
        # )
        # self.photos_rect = (
        #     self.photos[0].get_rect().move([0, 22]),
        #     self.photos[1].get_rect().move([928, 22 + 4]),
        #     self.photos[2].get_rect().move([1258, 22 + 5])
        # )
        if self.display:  # if display is required
            self.screen = pygame.display.set_mode(self.screen_size)  # set screen size
        # self.delta_t = 1 / 10  # simulation timing interval
        self.df = self._get_data_frame()

    @staticmethod
    def _get_data_frame():
        file_name = '/Users/atcold/Scratch/I-80/'
        file_name += '/vehicle-trajectory-data/0500pm-0515pm/trajectories-0500-0515.txt'
        return pd.read_table(file_name, sep='\s+', header=None, names=(
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
            'Preceding Vehicle',
            'Following Vehicle',
            'Spacing',
            'Headway'
        ))

    def step(self, policy_action=None):

        df = self.df
        valid_x = (df['Local Y'] * FOOT * SCALE - X_OFFSET).between(0, self.screen_size[0])
        vehicles = set(df[(df['Frame ID'] == self.frame) & valid_x]['Vehicle ID'])
        current_vehicles = set(v.id for v in self.vehicles)
        for vehicle_id in vehicles:
            if vehicle_id not in current_vehicles:
                car = self.EnvCar(df[(df['Vehicle ID'] == vehicle_id) & valid_x], self.offset)
                self.vehicles.append(car)
        for v in self.vehicles[:]:
            if v.off_screen:
                self.vehicles.remove(v)

        for v in self.vehicles:
            v.step(None)

        self.frame += 1

        return None, None, None, None

    def _draw_lanes(self, mode='human'):
        if mode == 'human':
            lanes = self.lanes  # lanes
            s = self.screen  # screen
            draw_line = pygame.draw.line  # shortcut
            w = colours['w']  # colour white
            sw = self.screen_size[0]  # screen width

            for lane in self.lanes:
                draw_dashed_line(s, w, (0, lane['min']), (sw, lane['min']), 3)
                draw_dashed_line(s, colours['r'], (0, lane['mid']), (sw, lane['mid']))

            draw_line(s, w, (0, lanes[0]['min']), (sw, lanes[0]['min']), 3)
            bottom = lanes[-1]['max']
            draw_line(s, w, (0, bottom), (18 * LANE_W, bottom), 3)
            draw_line(s, w, (0, bottom + 29), (18 * LANE_W, bottom + 29 - 0.035 * 18 * LANE_W), 3)
            draw_dashed_line(s, w, (18 * LANE_W, bottom + 13), (31 * LANE_W, bottom), 3)
            sw *= .9
            draw_dashed_line(s, colours['r'], (0, bottom + 42), (sw, bottom + 42 - 0.035 * sw))
            draw_line(s, w, (0, bottom + 53), (sw, bottom + 53 - 0.035 * sw), 3)
            draw_line(s, w, (sw, bottom + 3), (self.screen_size[0], bottom + 2), 3)
