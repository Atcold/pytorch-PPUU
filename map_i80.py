# from os import getpid, system
from os.path import isfile

import torch
from random import choice, randrange

from custom_graphics import draw_dashed_line
from traffic_gym import Simulator, Car, colours
import pygame
import pandas as pd
import numpy as np
import pdb, random
import bisect
import pdb, pickle, os, re

# Conversion LANE_W from real world to pixels
# A US highway lane width is 3.7 metres, here 50 pixels
LANE_W = 24  # pixels / 3.7 m, lane width
SCALE = LANE_W / 3.7  # pixels per metre
FOOT = 0.3048  # metres per foot
X_OFFSET = 470  # horizontal offset (camera 2 leftmost view)
MAX_SPEED = 130


class I80Car(Car):
    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W
    X_OFFSET = X_OFFSET
    max_a = 40
    max_b = 0.01

    def __init__(self, df, y_offset, look_ahead, screen_w, font=None, kernel=0, dt=1/10):
        k = kernel  # running window size
        self._length = df.at[df.index[0], 'Vehicle Length'] * FOOT * SCALE
        self._width = df.at[df.index[0], 'Vehicle Width'] * FOOT * SCALE
        self.id = df.at[df.index[0], 'Vehicle ID']  # extract scalar <'Vehicle ID'> <at> <index[0]>

        # X and Y are swapped in the I-80 data set...
        x = df['Local Y'].rolling(window=k).mean().shift(1 - k).values * FOOT * SCALE - self.X_OFFSET - self._length
        y = df['Local X'].rolling(window=k).mean().shift(1 - k).values * FOOT * SCALE + y_offset
        if dt > 1 / 10:
            s = int(dt * 10)
            end = len(x) - len(x) % s
            x = x[:end].reshape(-1, s).mean(axis=1)
            y = y[:end].reshape(-1, s).mean(axis=1)
        self._max_t = len(x) - np.count_nonzero(np.isnan(x)) - 2  # 2 for computing the acceleration

        self._trajectory = np.column_stack((x, y))
        self._position = self._trajectory[0]
        self._df = df
        self._frame = 0
        self._dt = dt
        # self._direction = np.array((1, 0), np.float)  # assumes horizontal if initially unknown
        self._direction = self._get('init_direction', 0)
        self._speed = self._get('speed', 0)
        self._colour = colours['c']
        self._braked = False
        self.off_screen = self._max_t <= 0
        self._states = list()
        self._states_image = list()
        self._ego_car_image = None
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
        self._lane_list = df['Lane Identification'].values
        self.collisions_per_frame = 0

    @property
    def is_autonomous(self):
        return False

    def _get(self, what, k):
        direction_vector = self._trajectory[k + 1] - self._trajectory[k]
        norm = np.linalg.norm(direction_vector)
        if what == 'direction':
            if norm < 1e-6: return self._direction  # if static returns previous direction
            return direction_vector / norm
        if what == 'speed':
            return norm / self._dt
        if what == 'init_direction':  # valid direction can be computed when speed is non-zero
            t = 1  # check if the car is in motion the next step
            while self._df.at[self._df.index[t], 'Vehicle Velocity'] < 5 and t < self._max_t: t += 1
            # t point to the point in time where speed is > 5
            direction_vector = self._trajectory[t] - self._trajectory[t - 1]
            norm = np.linalg.norm(direction_vector)
            # assert norm > 1e-6, f'norm: {norm} -> too small!'
            if norm < 1e-6:
                print(f'{self} has undefined direction, assuming horizontal')
                return np.array((1, 0), dtype=np.float)
            return direction_vector / norm

    # This was trajectories replay (to be used as ground truth, without any policy and action generation)
    # def step(self, action):
    #     position = self._position
    #     self._position = self._trajectory[self._frame]
    #     new_direction = self._position - position
    #     self._direction = new_direction if np.linalg.norm(new_direction) > 0.1 else self._direction
    #     self._direction /= np.linalg.norm(self._direction)
    #     assert 0.99 < np.linalg.norm(self._direction) < 1.01
    #     assert self._direction[0] > 0

    def policy(self, *args, **kwargs):
        self._frame += 1
        self.off_screen = self._frame >= self._max_t

        new_speed = self._get('speed', self._frame)
        a = (new_speed - self._speed) / self._dt

        ortho_direction = np.array((self._direction[1], -self._direction[0]))
        new_direction = self._get('direction', self._frame)
        b = (new_direction - self._direction).dot(ortho_direction) / (self._speed * self._dt + 1e-6)
        # if abs(b) > self._speed:
        #     b = self._speed * np.sign(b)

        # From an analysis of the action histograms -> limit a, b to sensible range
        a, b = self.action_clipping(a, b)

        # # Colour code for identifying trajectory divergence
        # measurement = self._trajectory[self._frame]
        # current_position = self._position
        # distance = min(np.linalg.norm(current_position - measurement) / (2 * LANE_W) * 255, 255)
        # self._colour = (distance, 255 - distance, 0)

        return np.array((a, b))

    def action_clipping(self, a, b):
        max_a = self.max_a
        max_b = self.max_b * min((25 / self._length) ** 2, 1)
        a = a if abs(a) < max_a else np.sign(a) * max_a
        b = b if abs(b) < max_b else np.sign(b) * max_b
        return a, b

    @property
    def current_lane(self):
        # 1: left-most, 6: right-most, 7: ramp
        return self._lane_list[self._frame] - 1

    def count_collisions(self, state):
        self.collisions_per_frame = 0
        # alpha = 1 * self.SCALE  # 1 m overlap collision
        # for cars in state:
        #     if cars:
        #         behind, ahead = cars
        #         if behind:
        #             d = self - behind
        #             if d[0] < -alpha and abs(d[1]) + alpha < (self._width + behind._width) / 2:
        #                 self.collisions_per_frame += 1
        #                 # print(f'Collision {self.collisions_per_frame}/6, behind, vehicle {behind.id}')
        #         if ahead:
        #             d = ahead - self
        #             if d[0] < -alpha and abs(d[1]) + alpha < (self._width + ahead._width) / 2:
        #                 self.collisions_per_frame += 1
        #                 # print(f'Collision {self.collisions_per_frame}/6, ahead, vehicle {ahead.id}')

        beta = 0.99
        if self._states_image and self._states_image[-1][2] > beta:
            self.collisions_per_frame += 1
            # print(f'Collision registered for vehicle {self}')
            # print(f'Accident! Check vehicle {self}. Proximity of {self._states_image[-1][2]}.')


class I80(Simulator):
    # Environment's car class
    EnvCar = I80Car

    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W
    X_OFFSET = X_OFFSET
    DUMP_NAME = 'data_i80_v0'

    def __init__(self, **kwargs):
        kwargs['nb_lanes'] = 6

        delta_t = kwargs['delta_t']
        assert delta_t >= 1 / 10, f'Minimum delta t is 0.1s > {delta_t:.2f}s you tried to set'
        assert (delta_t * 10).is_integer(), f'dt: {delta_t:.2f}s must be a multiple of 0.1s'

        super().__init__(**kwargs)

        self.screen_size = (85 * self.LANE_W, self.nb_lanes * self.LANE_W + 5 * self.LANE_W)
        # self.photos = (
        #     pygame.image.load('I-80/cam2.png'),
        #     pygame.image.load('I-80/cam3.png'),
        #     pygame.image.load('I-80/cam4.png'),
        #     pygame.image.load('I-80/cam5.png'),
        # )
        # self.photos_rect = (
        #     self.photos[0].get_rect().move([0, 22]),
        #     self.photos[1].get_rect().move([932, 22 + 2]),
        #     self.photos[2].get_rect().move([932 + 340, 22 + 2]),
        #     self.photos[3].get_rect().move([932 + 340 + 360, 22 - 2]),
        # )
        if self.display:  # if display is required
            self.screen = pygame.display.set_mode(self.screen_size)  # set screen size
        # self.delta_t = 1 / 10  # simulation timing interval
        self._time_slots = (
            'i80/trajectories-0400-0415',
            'i80/trajectories-0500-0515',
            'i80/trajectories-0515-0530',
        )
        self._t_slot = None
        self._black_list = {
            self._time_slots[0]:
                {1628, 2089, 2834, 2818, 2874,  # ground truth errors (GTE)
                 1383, 1430, 1456, 1589, 1913},  # kinematic modelling errors (KME)
            self._time_slots[1]:
                {537, 1119, 1261, 1215, 1288, 1381, 1382, 1348, 2512, 2462, 2442, 2427,
                 2407, 2486, 2296, 2427, 2552, 2500, 2616, 2555, 2586, 2669,
                 876, 882, 953, 1290, 1574, 2053, 2054, 2134, 2332, 2117, 2301, 2488,  # KME
                 2519, 2421, 2788},  # KME
            self._time_slots[2]:
                {269, 567, 722, 790, 860, 1603, 1651, 1734, 1762, 1734,
                 1800, 1722, 1878, 2056, 2075, 2258, 2252, 2285, 2362,
                 3004, 401, 510, 682, 680, 815, 827, 1675, 1780, 1751, 1831,  # KME
                 2200, 2080, 2119, 2170, 2369, 2480, 1797},  # KME
        }
        self.df = None
        self.vehicles_history = None
        self.lane_occupancy = None
        self.nb_lanes = 7
        self.smoothing_window = 15
        self.max_frame = -1
        pth = 'traffic-data/state-action-cost/data_i80_v0/data_stats.pth'
        self.data_stats = torch.load(pth) if self.normalise_state or self.normalise_action else None
        self.cached_data_frames = dict()
        self.episode = 0
        self.train_indx = None
        self.indx_order = None

    def _get_data_frame(self, time_slot, x_max, x_offset):
        if time_slot in self.cached_data_frames:
            return self.cached_data_frames[time_slot]
        file_name = f'traffic-data/xy-trajectories/{time_slot}'
        if isfile(file_name + '.pkl'):
            file_name += '.pkl'
            print(f'Loading trajectories from {file_name}')
            df = pd.read_pickle(file_name)
        elif isfile(file_name + '.txt'):
            file_name += '.txt'
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
                'Preceding Vehicle',
                'Following Vehicle',
                'Spacing',
                'Headway'
            ))
        else:
            raise FileNotFoundError(f'{file_name}.{{pkl,txt}} not found.')

        # Get valid x coordinate rows
        valid_x = (df['Local Y'] * FOOT * SCALE - x_offset).between(0, x_max)

        # Cache data frame for later retrieval
        self.cached_data_frames[time_slot] = df[valid_x]

        # Restrict data frame to valid x coordinates
        return df[valid_x]

    def _get_first_frame(self, v_id):
        vehicle_data = self.df[self.df['Vehicle ID'] == v_id]
        frame = vehicle_data.at[vehicle_data.index[0], 'Frame ID']
        return frame

    def reset(self, frame=None, time_slot=None, vehicle_id=None, train_only=False):

        # train_only = True  # uncomment this if doing RL, to set as default behaviour
        if train_only:
            ################################################################################
            # Looping over training split ONLY
            ################################################################################
            if self.train_indx is None:
                train_indx_file = '/home/atcold/Work/GitHub/pytorch-Traffic-Simulator/train_indx.pkl'
                if not os.path.isfile(train_indx_file):
                    import get_data_idx
                print('Loading training indices')
                with open(train_indx_file, 'rb') as f:
                    self.train_indx = pickle.load(f)
                self.indx_order = list(self.train_indx.keys())
                self.random.shuffle(self.indx_order)
            assert not(frame or time_slot or vehicle_id), 'Already selecting training episode from file.'
            time_slot, vehicle_id = self.train_indx[self.indx_order[self.episode % len(self.indx_order)]]
            self.episode += 1
            ################################################################################

        super().reset(control=(frame is None))
        # print(f'\n > Env on process {os.getpid()} is resetting')
        self._t_slot = self._time_slots[time_slot] if time_slot is not None else self.random.choice(self._time_slots)
        self.df = self._get_data_frame(self._t_slot, self.screen_size[0], self.X_OFFSET)
        self.max_frame = max(self.df['Frame ID'])
        if vehicle_id: frame = self._get_first_frame(vehicle_id)
        if frame is None:  # controlled
            # Start at a random valid (new_vehicles is not empty) initial frame
            frame_df = self.df['Frame ID'].values
            new_vehicles = set()
            while not new_vehicles:
                frame = self.random.randrange(min(frame_df), max(frame_df))
                vehicles_history = set(self.df[self.df['Frame ID'] <= frame]['Vehicle ID'])
                new_vehicles = set(self.df[self.df['Frame ID'] > frame]['Vehicle ID']) - vehicles_history
                new_vehicles -= self._black_list[self._t_slot]  # clean up fuckers
        if self.controlled_car:
            self.controlled_car['frame'] = frame
            self.controlled_car['v_id'] = vehicle_id
        self.frame = frame - int(self.delta_t * 10)
        self.vehicles_history = set()
        # # Account for off-track vehicles
        # with open('off_track.pkl', 'rb') as f:
        #     self.off_track = pickle.load(f)
        # self.off_track = set()
        # accident_file = '/Volumes/MyBox/home/atcold/Traffic/scripts/log/peach-pass-1/peach_ts1.out'
        # self.accident_file = open(accident_file)
        # self.accident = self.get_next_accident()
        # while self.accident['frame'] < frame: self.accident = self.get_next_accident()

    # def get_next_accident(self):
    #     file = self.accident_file
    #     line = file.readline()
    #     # Skip good frames
    #     a = 'Accident!'
    #     while not re.search(a, line or a): line = file.readline()
    #     frame = int(re.search('t=(\d+)', line).group(1)) if line else -1
    #     # Get all cars
    #     cars = list()
    #     while re.search(a, line):
    #         cars.append(int(re.search('Car\.(\d+)', line).group(1)))
    #         line = file.readline()
    #     return {
    #         'frame': frame,
    #         'cars': cars,
    #     }

    def step(self, policy_action=None):

        assert not self.done, 'Trying to step on an exhausted environment!'

        if self.normalise_action and policy_action is not None:
            np.multiply(policy_action, self.data_stats['a_std'], policy_action)  # multiply by the std
            np.add(policy_action, self.data_stats['a_mean'], policy_action)  # add the mean

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
                car = self.EnvCar(car_df, self.offset, self.look_ahead, self.screen_size[0], f, self.smoothing_window,
                                  dt=self.delta_t)
                self.vehicles.append(car)
                if self.controlled_car and \
                        not self.controlled_car['locked'] and \
                        self.frame >= self.controlled_car['frame'] and \
                        (self.controlled_car['v_id'] is None or vehicle_id == self.controlled_car['v_id']):
                    self.controlled_car['locked'] = car
                    car.is_controlled = True
                    car.buffer_size = self.nb_states
                    car.lanes = self.lanes
                    car.look_ahead = self.look_ahead
                    # print(f'Controlling car {car.id}')
                    # self.dump_folder = f'{self._t_slot}_{car.id}'
                    # print(f'Creating folder {self.dump_folder}')
                    # system(f'mkdir -p screen-dumps/{self.dump_folder}')
                    if self.store_sim_video:
                        self.ghost = self.EnvCar(car_df, self.offset, self.look_ahead, self.screen_size[0], f,
                                                 self.smoothing_window, dt=self.delta_t)
            self.vehicles_history |= vehicles  # union set operation

        self.lane_occupancy = [[] for _ in range(7)]
        if self.show_frame_count:
            print(f'\r[t={self.frame}]', end='')

        for v in self.vehicles[:]:
            if v.off_screen:
                # print(f'vehicle {v.id} [off screen]')
                if self.state_image and self.store:
                    file_name = os.path.join(self.data_dir, self.DUMP_NAME, os.path.basename(self._t_slot))
                    print(f'[dumping {v} in {file_name}]')
                    v.dump_state_image(file_name, 'tensor')
                self.vehicles.remove(v)
            else:
                # Insort it in my vehicle list
                lane_idx = v.current_lane
                assert v.current_lane < self.nb_lanes, f'{v} is in lane {v.current_lane} at frame {self.frame}'
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
            action = v.policy() if not v.is_autonomous else policy_action

            # Perform such action
            v.step(action)

            # Store state and action pair
            if (self.store or v.is_controlled) and v.valid:
                v.store('state', state)
                v.store('action', action)

            if v.is_controlled and v.valid:
                v.count_collisions(state)
                if v.collisions_per_frame > 0: self.collision = True

            # # Create set of off track vehicles
            # if v._colour[0] > 128:  # one lane away
            #     if v.id not in self.off_track:
            #         print(f'Adding {v} to off_track set and saving it to disk')
            #         self.off_track.add(v.id)
            #         with open('off_track.pkl', 'wb') as f:
            #             pickle.dump(self.off_track, f)

            # # Point out accidents (as in tracking bugs) in original trajectories
            # if self.frame == self.accident['frame']:
            #     if v.id in self.accident['cars']:
            #         v.collisions_per_frame = 1
            #         self.collision = True

        # if self.frame == self.accident['frame']:
        #     print('Colliding vehicles:', self.accident['cars'])
        #     self.accident = self.get_next_accident()

        # Keep the ghost updated
        if self.store_sim_video:
            if self.ghost and self.ghost.off_screen: self.ghost = None
            if self.ghost: self.ghost.step(self.ghost.policy())

        self.frame += int(self.delta_t * 10)

        # Run out of frames?
        self.done = self.frame >= self.max_frame or self.user_is_done

        if self.controlled_car and self.controlled_car['locked']:
            return_ = self.controlled_car['locked'].get_last(
                n=self.nb_states,
                done=self.done,
                norm_state=self.normalise_state and self.data_stats,
                return_reward=self.return_reward,
                gamma=self.gamma,
            )
            if return_: return return_

        # return observation, reward, done, info
        return None, None, self.done, None

    def _draw_lanes(self, surface, mode='human', offset=0):

        slope = 0.035

        lanes = self.lanes  # lanes

        if mode == 'human':
            s = surface  # screen
            draw_line = pygame.draw.line  # shortcut
            w = colours['w']  # colour white
            g = (128, 128, 128)
            sw = self.screen_size[0]  # screen width

            for lane in lanes:
                draw_line(s, g, (0, lane['min']), (sw, lane['min']), 1)
                # draw_dashed_line(s, colours['r'], (0, lane['mid']), (sw, lane['mid']))

            draw_line(s, w, (0, lanes[0]['min']), (sw, lanes[0]['min']), 3)
            bottom = lanes[-1]['max']
            draw_line(s, w, (0, bottom), (18 * LANE_W, bottom), 3)
            draw_line(s, w, (0, bottom + 29), (18 * LANE_W, bottom + 29 - slope * 18 * LANE_W), 3)
            draw_line(s, g, (18 * LANE_W, bottom + 13), (31 * LANE_W, bottom), 1)
            # draw_line(s, g, (0, bottom + 42), (60 * LANE_W, bottom + 42 - slope * 60 * LANE_W), 1)
            draw_line(s, w, (0, bottom + 53), (60 * LANE_W, bottom + 53 - slope * 60 * LANE_W), 3)
            draw_line(s, w, (60 * LANE_W, bottom + 3), (sw, bottom + 2), 3)

            look_ahead = MAX_SPEED * 1000 / 3600 * self.SCALE
            o = self.offset
            draw_line(s, (255, 255, 0), (look_ahead, o), (look_ahead, 9.4 * LANE_W))
            draw_line(s, (255, 255, 0), (sw - 1.75 * look_ahead, o), (sw - 1.75 * look_ahead, bottom))
            draw_line(s, (255, 255, 0), (sw - 0.75 * look_ahead, o), (sw - 0.75 * look_ahead, bottom), 5)

            # pygame.image.save(s, "i80-real.png")

        if mode == 'machine':
            s = surface  # screen
            draw_line = pygame.draw.line  # shortcut
            w = colours['r']  # colour white
            b = colours['b']  # colour blue
            sw = self.screen_size[0]  # screen width
            m = offset

            for lane in lanes:
                draw_line(s, w, (0, lane['min'] + m), (sw + 2 * m, lane['min'] + m), 1)

            bottom = lanes[-1]['max'] + m
            draw_line(s, w, (0, bottom), (m + 18 * LANE_W, bottom), 1)
            draw_line(s, w, (m, bottom + 29), (m + 18 * LANE_W, bottom + 29 - slope * 18 * LANE_W), 1)
            draw_line(s, w, (m + 18 * LANE_W, bottom + 13), (m + 31 * LANE_W, bottom), 1)
            draw_line(s, w, (m, bottom + 53), (m + 60 * LANE_W, bottom + 53 - slope * 60 * LANE_W), 1)
            draw_line(s, w, (m + 60 * LANE_W, bottom + 3), (2 * m + sw, bottom), 1)

            # offroad regions
            pygame.Surface.fill(s, b, pygame.Rect(m + 0, m + lanes[0]['min']-35, sw, 34))
            pygame.draw.polygon(s, b, [
                (m + 0, bottom+2),
                (m + 0, bottom + 29-1),
                (m + 18 * LANE_W, bottom + 29-1 - slope * 18 * LANE_W),
                (m + 18 * LANE_W, bottom+2)
            ])
            pygame.draw.polygon(s, b, [
                (m + 0, bottom + 54),
                (m + 0, bottom + 54+30),
                (m + 60 * LANE_W, bottom + 54+30),
                (m + 60 * LANE_W, bottom + 54 - slope * 60 * LANE_W)
            ])
            pygame.Surface.fill(
                s,
                b,
                pygame.Rect(m + 60 * LANE_W, bottom + 5, sw-60*LANE_W, 54+30-5))

            self._lane_surfaces[mode] = surface.copy()
            # pygame.image.save(surface, "i80-machine.png")
