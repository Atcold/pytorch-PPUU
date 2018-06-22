from random import choice, randrange

from custom_graphics import draw_dashed_line
from traffic_gym import StatefulEnv, Car, colours
import pygame
import pandas as pd
import numpy as np
import pdb, random
import bisect
import pdb, pickle, os
from ukfBicycle.ukf import Ukf

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

    def __init__(self, df, y_offset, look_ahead, screen_w, font=None, kernel=0):
        self._k = 0  # running window size
        self._length = df.at[df.index[0], 'Vehicle Length'] * FOOT * SCALE
        self._width = df.at[df.index[0], 'Vehicle Width'] * FOOT * SCALE
        self.id = df.at[df.index[0], 'Vehicle ID']  # extract scalar <'Vehicle ID'> <at> <index[0]>

        # X and Y are swapped in the I-80 data set...
        # x = df['Local Y'].rolling(window=self._k).mean().shift(1 - self._k).values * FOOT * SCALE - X_OFFSET - self._length
        # y = df['Local X'].rolling(window=self._k).mean().shift(1 - self._k).values * FOOT * SCALE + y_offset
        # for t in range(len(y) - 1):
        #     delta_x = x[t + 1] - x[t]
        #     delta_y = y[t + 1] - y[t]
        #     if abs(delta_y) > abs(delta_x) * 0.1:
        #         y[t + 1] = y[t] + np.sign(delta_y) * abs(delta_x) * 0.1
        x = df['Local Y'].values
        y = df['Local X'].values

        self._trajectory = np.column_stack((x, y))
        self._position = self._trajectory[0]
        self._df = df
        self._frame = 0
        self._dt = 1 / 10
        self._direction = np.array((1, 0), np.float)  # assumes horizontal if initially unknown
        self._direction = self._get('direction', 0)
        self._speed = self._get('speed', 0)
        self._colour = colours['c']
        self._braked = False
        self.off_screen = len(df) <= self._k + 1
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
        if font is not None:
            self._text = self.get_text(self.id, font)
        self.is_controlled = False
        self.ukf = Ukf(
            dt=self._dt, startx=x[0], starty=y[0], startspeed=self._speed, starthdg=np.arctan2(*self._direction[::-1]),
            noise=1e-4, wheelbase=df.at[df.index[0], 'Vehicle Length'] * .7
        )
        self._y_offset = y_offset
        # if self.id == 36:
        #     print(self.ukf.x)
        #     print('Saving trajectory to file')
        #     with open('v35.pickle', 'wb') as f: pickle.dump(self._trajectory, f)

    @property
    def is_autonomous(self):
        return False

    def _get(self, what, k):
        direction_vector = self._trajectory[k + 1] - self._trajectory[k]  # TODO: use my own coordinates!!!
        norm = np.linalg.norm(direction_vector)
        if what == 'direction':
            if norm < 1e-6: return self._direction  # if static returns previous direction
            return direction_vector / norm
        if what == 'speed':
            return norm / self._dt

    # This was trajectories replay (to be used as ground truth, without any policy and action generation)
    # def step(self, action):
    #     position = self._position
    #     self._position = self._trajectory[self._frame]
    #     new_direction = self._position - position
    #     self._direction = new_direction if np.linalg.norm(new_direction) > 0.1 else self._direction
    #     self._direction /= np.linalg.norm(self._direction)
    #     assert 0.99 < np.linalg.norm(self._direction) < 1.01
    #     assert self._direction[0] > 0

    def step(self, action):
        self._frame += 1
        self.ukf.step(self._trajectory[self._frame])
        # if self.id == 36:
        #     print('f:', self._frame, 'x, y:', self._trajectory[self._frame])
        #     print(self.ukf.x)
        x = self.ukf.ukf.x[0] * FOOT * SCALE - X_OFFSET - self._length
        y = self.ukf.ukf.x[1] * FOOT * SCALE + self._y_offset
        self._position = np.array((x, y))
        self._speed = self.ukf.ukf.x[2] * FOOT * SCALE
        self._direction = np.array((np.cos(self.ukf.ukf.x[3]), np.sin(self.ukf.ukf.x[3])))

    def policy(self, observation, **kwargs):
        # self._frame += 1
        self.off_screen = self._frame >= len(self._df) - self._k - 2

        new_speed = self._get('speed', self._frame)
        a = (new_speed - self._speed) / self._dt

        ortho_direction = np.array((self._direction[1], -self._direction[0]))
        new_direction = self._get('direction', self._frame)
        b = (new_direction - self._direction).dot(ortho_direction) / (self._speed * self._dt + 1e-6)
        if abs(b) > self._speed:
            b = self._speed * np.sign(b)
        # return np.array((a, b))
        return np.zeros((2,))

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
        self._time_slots = (
            './data_i80/trajectories-0400-0415',
            './data_i80/trajectories-0500-0515',
            './data_i80/trajectories-0515-0530',
        )
        self._t_slot = None
        self._black_list = {
            self._time_slots[0]:
                {1628, 2089, 2834, 2818, 2874},
            self._time_slots[1]:
                {537, 1119, 1261, 1215, 1288, 1381, 1382, 1348, 2512, 2462, 2442, 2427,
                 2407, 2486, 2296, 2427, 2552, 2500, 2616, 2555, 2586, 2669},
            self._time_slots[2]:
                {269, 567, 722, 790, 860, 1603, 1651, 1734, 1762, 1734,
                 1800, 1722, 1878, 2056, 2075, 2258, 2252, 2285, 2362},
        }
        self.df = None
        self.vehicles_history = None
        self.lane_occupancy = None
        self._lane_surfaces = dict()
        self.nb_lanes = 7
        self.smoothing_window = 15
        self.random = random.Random()
        self.random.seed(54321)

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

    def reset(self, frame=None, time_slot=None):
        super().reset(control=(frame is None))
        self._t_slot = self._time_slots[time_slot] if time_slot is not None else choice(self._time_slots)
        self.df = self._get_data_frame(self._t_slot + '.txt', self.screen_size[0])
        if frame is None:  # controlled
            # Start at a random valid (new_vehicles is not empty) initial frame
            frame_df = self.df['Frame ID'].values
            new_vehicles = set()
            while not new_vehicles:
                frame = self.random.randrange(min(frame_df), max(frame_df))
                vehicles_history = set(self.df[self.df['Frame ID'] <= frame]['Vehicle ID'])
                new_vehicles = set(self.df[self.df['Frame ID'] > frame]['Vehicle ID']) - vehicles_history
                new_vehicles -= self._black_list[self._t_slot]  # clean up fuckers
        self.frame = frame
        self.vehicles_history = set()

    def step(self, policy_action=None):

        df = self.df
        now = df['Frame ID'] == self.frame
        vehicles = set(df[now]['Vehicle ID']) - self.vehicles_history - self._black_list[self._t_slot]

        if vehicles:
            now_and_on = df['Frame ID'] >= self.frame
            for vehicle_id in vehicles:
                this_vehicle = df['Vehicle ID'] == vehicle_id
                car_df = df[this_vehicle & now_and_on]
                if len(car_df) < self.smoothing_window + 1: continue
                f = self.font[20] if self.display else None
                car = self.EnvCar(car_df, self.offset, self.look_ahead, self.screen_size[0], f, self.smoothing_window)
                self.vehicles.append(car)
                if self.controlled_car and \
                        not self.controlled_car['locked'] and \
                        self.vehicles_history:
                    self.controlled_car['locked'] = car
                    car.is_controlled = True
                    car.buffer_size = self.nb_states
                    car.lanes = self.lanes
                    car.look_ahead = self.look_ahead
                    print('Controlling car {}'.format(car.id))
            self.vehicles_history |= vehicles  # union set operation

        self.lane_occupancy = [[] for _ in range(7)]
        print('\r[t={}]'.format(self.frame), end='')

        for v in self.vehicles[:]:
            if v.off_screen:
                # print(f'vehicle {v.id} [off screen]')
                if self.state_image and self.store:
                    file_name = os.path.join('scratch/data_i80_v4/', os.path.basename(self._t_slot))
                    print('[dumping {}]'.format(file_name))
                    v.dump_state_image(file_name, 'tensor')
                self.vehicles.remove(v)
            else:
                # Insort it in my vehicle list
                lane_idx = v.current_lane
                bisect.insort(self.lane_occupancy[lane_idx], v)

        if self.state_image or self.controlled_car and self.controlled_car['locked']:
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
            action = v.policy(state) if not v.is_autonomous else policy_action

            # Perform such action
            v.step(action)

            # Store state and action pair
            if (self.store or v.is_controlled) and v.valid:
                v.store('state', state)
                v.store('action', action)

            if v.is_controlled and v.valid:
                v.count_collisions(state)
                if v.collisions_per_frame > 0: self.collision = True

        self.frame += 1

        # return observation, reward, done, info

        if self.controlled_car and self.controlled_car['locked']:
            return_ = self.controlled_car['locked'].get_last(self.nb_states)
            if return_: return return_

        return None, None, False, None

    def _draw_lanes(self, surface, mode='human', offset=0):

        slope = 0.035

        lanes = self.lanes  # lanes

        if mode == 'human':
            s = surface  # screen
            draw_line = pygame.draw.line  # shortcut
            w = colours['w']  # colour white
            sw = self.screen_size[0]  # screen width

            for lane in lanes:
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
            s = surface  # screen
            draw_line = pygame.draw.line  # shortcut
            w = colours['r']  # colour white
            sw = self.screen_size[0]  # screen width
            m = offset

            for lane in lanes:
                draw_line(s, w, (0, lane['min'] + m), (sw + 2 * m, lane['min'] + m), 1)

            draw_line(s, w, (0, lanes[0]['min'] + m), (sw, lanes[0]['min'] + m), 1)
            bottom = lanes[-1]['max'] + m
            draw_line(s, w, (0, bottom), (m + 18 * LANE_W, bottom), 1)
            draw_line(s, w, (m, bottom + 29), (m + 18 * LANE_W, bottom + 29 - slope * 18 * LANE_W), 1)
            draw_line(s, w, (m + 18 * LANE_W, bottom + 13), (m + 31 * LANE_W, bottom), 1)
            draw_line(s, w, (m, bottom + 53), (m + 60 * LANE_W, bottom + 53 - slope * 60 * LANE_W), 1)
            draw_line(s, w, (m + 60 * LANE_W, bottom + 3), (2 * m + sw, bottom), 1)

            self._lane_surfaces[mode] = surface.copy()
