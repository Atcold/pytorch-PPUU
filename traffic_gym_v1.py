from custom_graphics import draw_dashed_line
from traffic_gym import StatefulEnv, Car, colours
import pygame
import pandas as pd
import numpy as np
import pdb
import bisect
import pdb, pickle, os

# Conversion LANE_W from real world to pixels
# A US highway lane width is 3.7 metres, here 50 pixels
LANE_W = 24  # pixels / 3.7 m, lane width
SCALE = LANE_W / 3.7  # pixels per metre
FOOT = 0.3048  # metres per foot
X_OFFSET = 470  # horizontal offset (camera 2 leftmost view)
MAX_SPEED = 130


class RealCar(Car):
    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W

    def __init__(self, df, y_offset, look_ahead, screen_w, font):
        self._k = 15  # running window size
        # self._k = 0
        self._length = df.at[df.index[0], 'Vehicle Length'] * FOOT * SCALE
        self._width = df.at[df.index[0], 'Vehicle Width'] * FOOT * SCALE
        self.id = df.at[df.index[0], 'Vehicle ID']  # extract scalar <'Vehicle ID'> <at> <index[0]>

        # X and Y are swapped in the I-80 data set...
        x = df['Local Y'].rolling(window=self._k).mean().shift(1 - self._k).values * FOOT * SCALE - X_OFFSET - self._length
        y = df['Local X'].rolling(window=self._k).mean().shift(1 - self._k).values * FOOT * SCALE + y_offset
        for t in range(len(y) - 1):
            delta_x = x[t + 1] - x[t]
            delta_y = y[t + 1] - y[t]
            if abs(delta_y) > abs(delta_x) * 0.1:
                y[t + 1] = y[t] + np.sign(delta_y) * abs(delta_x) * 0.1

        self._trajectory = np.column_stack((x, y))
        self._position = self._trajectory[0]
        self._df = df
        self._frame = 0
        self._dt = 1 / 10
        self._direction = self._get('direction', 0)
        self._speed = self._get('speed', 0)
        self._colour = colours['c']
        self._braked = False
        self.off_screen = False
        self._states = list()
        self._states_image = list()
        self._actions = list()
        self._passing = False
        self._actions = list()
        self._states = list()
        self.states_image = list()
        self.look_ahead = look_ahead
        self.screen_w = screen_w
        self._safe_factor = 1.5  # second, manually matching the data
        self._text = self.get_text(self.id, font)

    def _get(self, what, k):
        direction_vector = self._trajectory[k + 1] - self._trajectory[k]
        if what == 'direction':
            return direction_vector / (np.linalg.norm(direction_vector) + 1e-6)
        if what == 'speed':
            return np.linalg.norm(direction_vector) / self._dt

    # This was trajectories reply (to be used as ground truth, without any policy and action generation)
    # def step(self, action):
    #     self._frame += 1
    #     position = self._position
    #     df = self._trajectory
    #     self._position = df.loc[df.index[self._frame], ['x', 'y']].values
    #     new_direction = self._position - position
    #     self._direction = new_direction if np.linalg.norm(new_direction) > 1 else self._direction
    #     self._direction /= np.linalg.norm(self._direction)
    #     self.off_screen = self._frame >= len(df) - self._k

    def policy(self, observation):
        self._frame += 1
        self.off_screen = self._frame >= len(self._df) - self._k - 2

        new_speed = self._get('speed', self._frame)
        a = (new_speed - self._speed) / self._dt

        ortho_direction = np.array((self._direction[1], -self._direction[0]))
        new_direction = self._get('direction', self._frame)
        b = (new_direction - self._direction).dot(ortho_direction) / (self._speed * self._dt + 1e-6)

        return np.array((a, b))

    @property
    def current_lane(self):
        # 1: left-most, 6: right-most, 7: ramp
        return self._df.at[self._df.index[self._frame], 'Lane Identification'] - 1


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

        self.screen_size = (85 * self.LANE_W, self.nb_lanes * self.LANE_W + 5 * self.LANE_W)
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
        self.file_name = './data_i80/trajectories-0515-0530.txt'
#        self.file_name = './data_i80/trajectories-0500-0515.txt'
#        self.file_name = './data_i80/trajectories-0400-0415.txt'
        self.df = self._get_data_frame(self.file_name, self.screen_size[0])
        self.vehicles_history = set()
        self.lane_occupancy = None
        # self._kf = KalmanFilter(
        #     transition_matrices=np.array([[1, 1], [0, 1]]),
        #     transition_covariance=0.00001 * np.eye(2)
        # ).em(self.df[self.df['Vehicle ID'] == self.df.at[self.df.index[0], 'Vehicle ID']].loc[:, ['Local X']].values)
        self._lane_surfaces = dict()

    @staticmethod
    def _get_data_frame(file_name, x_max):
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
            'Preceding Vehicle',
            'Following Vehicle',
            'Spacing',
            'Headway'
        ))

        # Get valid x coordinate rows
        valid_x = (df['Local Y'] * FOOT * SCALE - X_OFFSET).between(0, x_max)

        # Restrict data frame to valid x coordinates
        return df[valid_x]

    def step(self, policy_action=None):

        df = self.df
        now = df['Frame ID'] == self.frame
        vehicles = set(df[now]['Vehicle ID']) - self.vehicles_history

        if vehicles:
            now_and_on = df['Frame ID'] >= self.frame
            for vehicle_id in vehicles:
                this_vehicle = df['Vehicle ID'] == vehicle_id
                car = self.EnvCar(df[this_vehicle & now_and_on], self.offset, self.look_ahead, self.screen_size[0], self.font[20])
                self.vehicles.append(car)
            self.vehicles_history |= vehicles  # union set operation

        self.lane_occupancy = [[] for _ in range(7)]
        print('[t={}]'.format(self.frame), end="\r")

        for v in self.vehicles[:]:
            if v.off_screen:
#                print(f'vehicle {v.id} [off screen]')
                if self.state_image and self.store:
                    file_name = os.path.join('scratch/data_i80_v2', os.path.basename(self.file_name))
#                    print(f'[dumping {file_name}]')
                    v.dump_state_image(file_name, 'tensor')
                self.vehicles.remove(v)
            else:
                # Insort it in my vehicle list
                lane_idx = v.current_lane
                bisect.insort(self.lane_occupancy[lane_idx], v)

        if self.state_image:
            # How much to look far ahead
            look_ahead = MAX_SPEED * 1000 / 3600 * self.SCALE
            look_sideways = 2 * self.LANE_W
            self.render(mode='machine', width_height=(2 * look_ahead, 2 * look_sideways), scale=0.25)

        for v in self.vehicles:

            # Generate symbolic state
            lane_idx = v.current_lane
            left_vehicles = self._get_neighbours(lane_idx, -1, v) \
                if 0 < lane_idx < 6 or lane_idx == 6 and v.front[0] > 18 * LANE_W else None
            mid_vehicles = self._get_neighbours(lane_idx, 0, v)
            right_vehicles = self._get_neighbours(lane_idx, + 1, v) \
                if lane_idx < 5 or lane_idx == 5 and v.front[0] > 18 * LANE_W else None
            state = left_vehicles, mid_vehicles, right_vehicles

            # Sample an action based on the current state
            action = v.policy(state)

            # Perform such action
            v.step(action)

            # Store state and action pair
            if self.store and v.valid:
                v.store('state', state)
                v.store('action', action)

        self.frame += 1

        return None, None, None

    def _draw_lanes(self, surface, mode='human', offset=0):

        slope = 0.035
        if mode == 'human':
            lanes = self.lanes  # lanes
            s = surface  # screen
            draw_line = pygame.draw.line  # shortcut
            w = colours['w']  # colour white
            sw = self.screen_size[0]  # screen width

            for lane in self.lanes:
                draw_dashed_line(s, w, (0, lane['min']), (sw, lane['min']), 3)
                draw_dashed_line(s, colours['r'], (0, lane['mid']), (sw, lane['mid']))

            draw_line(s, w, (0, lanes[0]['min']), (sw, lanes[0]['min']), 3)
            bottom = lanes[-1]['max']
            draw_line(s, w, (0, bottom), (18 * LANE_W, bottom), 3)
            draw_line(s, w, (0, bottom + 29), (18 * LANE_W, bottom + 29 - slope * 18 * LANE_W), 3)
            draw_dashed_line(s, w, (18 * LANE_W, bottom + 13), (31 * LANE_W, bottom), 3)
            draw_dashed_line(s, colours['r'], (0, bottom + 42), (60 * LANE_W, bottom + 42 - slope * 60 * LANE_W))
            draw_line(s, w, (0, bottom + 53), (60 * LANE_W, bottom + 53 - slope * 60 * LANE_W), 3)
            draw_line(s, w, (60 * LANE_W, bottom + 3), (sw, bottom + 2), 3)

            look_ahead = MAX_SPEED * 1000 / 3600 * self.SCALE
            o = self.offset
            draw_line(s, (255, 255, 0), (look_ahead, o), (look_ahead, 9.4 * LANE_W))
            draw_line(s, (255, 255, 0), (sw - 1.75 * look_ahead, o), (sw - 1.75 * look_ahead, bottom))
            draw_line(s, (255, 255, 0), (sw - 0.75 * look_ahead, o), (sw - 0.75 * look_ahead, bottom), 5)

        if mode == 'machine':
            lanes = self.lanes  # lanes
            s = surface  # screen
            draw_line = pygame.draw.line  # shortcut
            w = colours['r']  # colour white
            sw = self.screen_size[0]  # screen width
            m = offset

            for lane in self.lanes:
                draw_line(s, w, (0, lane['min'] + m), (sw + 2 * m, lane['min'] + m), 1)

            draw_line(s, w, (0, lanes[0]['min'] + m), (sw, lanes[0]['min'] + m), 1)
            bottom = lanes[-1]['max'] + m
            draw_line(s, w, (0, bottom), (m + 18 * LANE_W, bottom), 1)
            draw_line(s, w, (m, bottom + 29), (m + 18 * LANE_W, bottom + 29 - slope * 18 * LANE_W), 1)
            draw_line(s, w, (m + 18 * LANE_W, bottom + 13), (m + 31 * LANE_W, bottom), 1)
            draw_line(s, w, (m, bottom + 53), (m + 60 * LANE_W, bottom + 53 - slope * 60 * LANE_W), 1)
            draw_line(s, w, (m + 60 * LANE_W, bottom + 3), (2 * m + sw, bottom), 1)

            self._lane_surfaces[mode] = surface.copy()
