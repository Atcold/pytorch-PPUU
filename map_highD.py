import bisect
import os
import pickle

import numpy as np
import pandas as pd
import pygame
import torch

from custom_graphics import draw_dashed_line
from traffic_gym import Simulator, Car, colours

SCALE = 5
Y_OFFSET = 50
MAX_SPEED = 130
DT = 1 / 25  # frame rate for highD dataset (comes from recording_meta csv files)


# Recording meta data
def read_recoding_meta(filename, reload=False):
    # Check if concatenated file already exists:
    if os.path.isfile(os.path.join(filename, 'all_recordingMeta.pkl')) and not reload:
        df_recs = pd.read_pickle(os.path.join(filename, 'all_recordingMeta.pkl'))
        return df_recs
    # Else read in the individual recordings, concatenate, and save them
    df_rec_dict = dict()
    recordings = [f'{i:02d}' for i in range(1, 61)]
    for rec in recordings:
        # Recording meta data dataframes
        df_rec = pd.read_csv(os.path.join(filename, f'{rec}_recordingMeta.csv'),
                             header=0,
                             index_col=False,
                             names=(
                                 'Recording ID',
                                 'Frame Rate',
                                 'Location ID',
                                 'Speed Limit',
                                 'Month',
                                 'Weekday',
                                 'Start Time',
                                 'Duration',
                                 'Total Driven Distance',
                                 'Total Driven Time',
                                 'Number Vehicles',
                                 'Number Cars',
                                 'Number Trucks',
                                 'Upper Lane Markings',
                                 'Lower Lane Markings'
                             ))
        # Calculate number of lanes based on lane markings
        num_lanes = len(df_rec["Upper Lane Markings"].values[0].split(";")) + \
                    len(df_rec["Lower Lane Markings"].values[0].split(";"))
        df_rec["Number Lanes"] = num_lanes - 2
        df_rec['Upper Lane Markings'] = df_rec['Upper Lane Markings'].apply(lambda x: np.fromstring(x, sep=';'))
        df_rec['Lower Lane Markings'] = df_rec['Lower Lane Markings'].apply(lambda x: np.fromstring(x, sep=';'))

        # Add to dict
        df_rec_dict[rec] = df_rec
    # Concatenate and save
    df_recs = pd.concat([df_rec_dict[rec] for rec in recordings], ignore_index=True)
    df_recs.to_pickle(os.path.join(filename, 'all_recordingMeta.pkl'))
    return df_recs


class HighDCar(Car):
    # Global constants
    SCALE = SCALE
    Y_OFFSET = Y_OFFSET
    max_a = 100   # TODO: Confirm with Alfredo that this number is ok
    max_b = 0.02  # TODO: Confirm with Alfredo that this number is ok

    def __init__(self, df, look_ahead, screen_w, font=None, dt=DT):
        self._driving_direction = df.at[df.index[0], 'Driving Direction']  # +1 := left-to-right; -1 := right-to-left
        self._length = df.at[df.index[0], 'Vehicle Length'] * SCALE
        self._width = df.at[df.index[0], 'Vehicle Width'] * SCALE
        self.id = df.at[df.index[0], 'Vehicle ID']  # extract scalar <'Vehicle ID'> <at> <index[0]>
        # X,Y position in highD dataset is the top left corner of the vehicles bounding box:
        #   subtract car length if vehicle is driving left to right so that x position is at the rear of the vehicle
        x = df['Local Offset X'].values*SCALE - (self._length if self._driving_direction < 0 else 0)
        y = df['Local Y'].values * SCALE + Y_OFFSET + (self._width / 2)  # place y position in middle of vehicle
        if dt > DT:
            s = int(dt / DT)
            end = len(x) - len(x) % s
            x = x[:end].reshape(-1, s).mean(axis=1)
            y = y[:end].reshape(-1, s).mean(axis=1)
        self._max_t = len(x) - int(2 * dt / DT) - (len(x) % (dt/DT))  # 2 for computing the acceleration

        self._trajectory = np.column_stack((x, y))
        self._position = self._trajectory[0]
        self._df = df
        self._upper_lane_markings = df.at[df.index[0], 'Upper Lane Markings'] * SCALE + Y_OFFSET
        self._lower_lane_markings = df.at[df.index[0], 'Lower Lane Markings'] * SCALE + Y_OFFSET
        self._frame = 0
        self._dt = dt
        self._direction = self._get('init_direction', 0)
        self._speed = self._get('speed', 0)
        self._colour = colours['c']
        self._braked = False
        self.off_screen = self._max_t <= 0
        self._states = list()
        self._states_image = list()
        self._actions = list()
        self._passing = False
        self._actions = list()
        self._states = list()
        self.states_image = list()
        self.look_ahead = look_ahead
        self.screen_w = screen_w
        self._safe_factor = 1.5  # second, manually matching the data TODO: Check this number
        if font is not None:
            self._text = self.get_text(self.id, font)
            # self._text = self.get_text(f'{self.id}: Lane {self.current_lane}', font)  # Uncomment to display lane #
        self.is_controlled = False
        self.collisions_per_frame = 0
        if self.id == 231:
            print(f'Initial calc speed: {self._speed / SCALE * 3.6:.2f} km/h', end=' ')
            print(f'Initial DF Speed: {df.at[df.index[0], "Vehicle Velocity"] * 3.6:.2f} km\h')

    @property
    def is_autonomous(self):
        return False

    def _get(self, what, k):
        direction_vector = self._trajectory[k + int(self._dt/DT)] - self._trajectory[k]
        norm = np.linalg.norm(direction_vector)
        if what == 'direction' or what == 'init_direction':
            assert norm > 1e-6, f'{self.id} is static at time step {k}! Speed: {norm}'
            return direction_vector / norm
        if what == 'speed':
            return norm / self._dt

    # # This was trajectories replay (to be used as ground truth, without any policy and action generation)
    # def step(self, action):
    #     position = self._position
    #     self._position = self._trajectory[self._frame]
    #     new_direction = self._position - position
    #     self._direction = new_direction if np.linalg.norm(new_direction) > 0.1 else self._direction
    #     self._direction /= np.linalg.norm(self._direction)
    #     assert 0.99 < np.linalg.norm(self._direction) < 1.01
    #     assert self._direction[0] * self._driving_direction > 0

    def policy(self, *args, **kwargs):
        self._frame += int(self._dt / DT)
        self.off_screen = self._frame >= self._max_t
        new_speed = self._get('speed', self._frame)
        a = (new_speed - self._speed) / self._dt
        if self.id == 231:
            df = self._df
            print(f' Calc Speed: {new_speed / SCALE * 3.6:.2f} km/h', end=' ')
            print(f'DF Speed: {df.at[df.index[self._frame], "Vehicle Velocity"] * 3.6:.2f} km\h', end=' ')
            print(f'Calc Acc: {a / SCALE / 9.81: .2f} g', end=' ')
            print(f'DF Acc: {df.at[df.index[self._frame], "Vehicle Acceleration"] / 9.81:.2f} g')

        ortho_direction = np.array((self._direction[1], -self._direction[0]))
        new_direction = self._get('direction', self._frame)
        b = (new_direction - self._direction).dot(ortho_direction) / (self._speed * self._dt + 1e-6)

        # From an analysis of the action histograms -> limit a, b to sensible range
        assert a / SCALE < self.max_a, f'Car {self.id} acceleration magnitude out of range: {a/SCALE} > {self.max_a}'
        assert b < self.max_b, f'Car {self.id} acceleration angle out of range: {b} > {self.max_b}'

        # Colour code for identifying trajectory divergence if not self.off_screen:
        lane_width = (self._upper_lane_markings[1] - self._upper_lane_markings[0])
        measurement = self._trajectory[self._frame]
        current_position = self._position
        distance = min(np.linalg.norm(current_position - measurement) / (2 * lane_width) * 255, 255)
        self._colour = (distance, 255 - distance, 0)
        return np.array((a, b))

    @property  # Lanes are 0 indexed
    def current_lane(self):
        current_y = self._trajectory[self._frame][1]
        lane = 0
        if self._driving_direction > 0:  # Check if in lower lanes (i.e. driving left-to-right)
            while current_y > self._lower_lane_markings[lane]:
                lane += 1
            return lane + len(self._upper_lane_markings) - 2
        # Else, in upper lanes (i.e. driving right-to-left)
        while current_y > self._upper_lane_markings[lane]:
            lane += 1
        return lane - 1

    def count_collisions(self, state):
        self.collisions_per_frame = 0
        alpha = 1 * SCALE  # 1 m overlap collision
        for cars in state:
            if cars:
                behind, ahead = cars
                if behind:
                    d = self - behind
                    if d[0] < -alpha and abs(d[1]) + alpha < (self._width + behind._width) / 2:
                        self.collisions_per_frame += 1
                        print(f'Collision {self.collisions_per_frame}/6, behind, vehicle {behind.id}')
                if ahead:
                    d = ahead - self
                    if d[0] < -alpha and abs(d[1]) + alpha < (self._width + ahead._width) / 2:
                        self.collisions_per_frame += 1
                        print(f'Collision {self.collisions_per_frame}/6, ahead, vehicle {ahead.id}')

        beta = 0.99
        if self._states_image and self._states_image[-1][2] > beta:
            self.collisions_per_frame += 1
            print(f'Collision registered for vehicle {self}')
            print(f'Accident! Check vehicle {self}. Proximity of {self._states_image[-1][2]}.')

    @property
    def valid(self):
        if self._driving_direction > 0:  # car is driving left to right
            return self.back[0] > self.look_ahead and self.front[0] < self.screen_w - self.look_ahead
        else:  # car is driving right to left
            return self.front[0] > self.look_ahead and self.back[0] < self.screen_w - self.look_ahead


class HighD(Simulator):
    # Environment's car class
    EnvCar = HighDCar

    # Global constants
    SCALE = SCALE
    Y_OFFSET = Y_OFFSET
    MAX_SPEED = MAX_SPEED
    DUMP_NAME = 'data_highD_v0'

    def __init__(self, **kwargs):
        self.rec_meta = read_recoding_meta('traffic-data/xy-trajectories/highD/', reload=True)
        kwargs['nb_lanes'] = self.rec_meta[self.rec_meta['Recording ID']
                                           == int(kwargs['rec'])]["Number Lanes"].values[0]
        self.recording = kwargs['rec']
        del kwargs['rec']
        delta_t = kwargs['delta_t']
        assert delta_t >= DT, f'Minimum delta t is 0.04s > {delta_t:.2f}s you tried to set'
        assert (delta_t / DT).is_integer(), f'dt: {delta_t:.2f}s must be a multiple of 1 / 25 s'

        super().__init__(**kwargs)

        self.screen_size = (1800, 400)
        # # Uncomment below to display actual image from recording
        # photo = pygame.image.load(f'HighD/{self.recording}_highway.png')
        # photo = pygame.transform.scale(photo, self.screen_size)
        # self.photos = (photo,)
        # self.photos_rect = (self.photos[0].get_rect().move([0, Y_OFFSET]),)
        if self.display:  # if display is required
            self.screen = pygame.display.set_mode(self.screen_size)  # set screen size
            self.font = {
                15: pygame.font.SysFont(None, 15),
                20: pygame.font.SysFont(None, 20),
                30: pygame.font.SysFont(None, 30),
            }
        self._time_slots = [self.recording]
        self._t_slot = None
        self._black_list = {i: set() for i in self._time_slots}  # TODO: Need to visually inspect data to create this
        self.df = None
        self.vehicles_history = None
        self.lane_occupancy = None
        self.x_offset = 0
        self.nb_lanes = kwargs['nb_lanes']
        self.lanes = self.build_lanes(kwargs['nb_lanes'])
        self.max_frame = -1
        pth = 'traffic-data/state-action-cost/data_highD_v0/data_stats.pth'
        self.data_stats = torch.load(pth) if self.normalise_state or self.normalise_action else None
        self.cached_data_frames = dict()
        self.episode = 0
        self.train_indx = None
        self.indx_order = None
        self.look_ahead = MAX_SPEED * 1000 / 3600 * self.SCALE  # km/h --> m/s * SCALE
        upper_lanes = self.rec_meta[self.rec_meta['Recording ID'] ==
                                    int(self.recording)]['Upper Lane Markings'].values[0]
        self.lane_width = (upper_lanes[1] - upper_lanes[0]) * SCALE  # distance between first and second lane markings

    def build_lanes(self, nb_lanes):
        upper_lanes = self.rec_meta[self.rec_meta['Recording ID'] ==
                                    int(self.recording)]["Upper Lane Markings"].values[0] * SCALE + Y_OFFSET
        lower_lanes = self.rec_meta[self.rec_meta['Recording ID'] ==
                                    int(self.recording)]["Lower Lane Markings"].values[0] * SCALE + Y_OFFSET

        lane_markings = [
            {'min': upper_lanes[i],
             'mid': (upper_lanes[i] + upper_lanes[i+1]) / 2,
             'max': upper_lanes[i+1]}
            for i in range(len(upper_lanes)-1)]
        for i in range(len(lower_lanes)-1):
            lane_markings.append(
                {'min': lower_lanes[i],
                 'mid': (lower_lanes[i] + lower_lanes[i+1]) / 2,
                 'max': lower_lanes[i+1]})
        return tuple(lane_markings)

    def _read_data_frame(self, file_name, rec):
        df_rec = self.rec_meta
        # Track meta data dataframes
        df_track_meta = pd.read_csv(os.path.join(file_name, f'{rec}_tracksMeta.csv'),
                                    header=0,
                                    names=(
                                        'Vehicle ID',
                                        'Vehicle Length_Meta',
                                        'Vehicle Width_Meta',
                                        'Initial Frame',
                                        'Final Frame',
                                        'Number Frames',
                                        'Vehicle Class',
                                        'Driving Direction',
                                        'Traveled Distance',
                                        'Min Vehicle Velocity X',
                                        'Max Vehicle Velocity X',
                                        'Mean Vehicle Velocity X',
                                        'Min Spacing',
                                        'Min Headway',
                                        'Min Time to Collision',
                                        'Number Lane Changes'
                                    ))
        # Convert driving direction to +1 for left-to-right and -1 for right-to-left
        df_track_meta['Driving Direction'] = df_track_meta['Driving Direction'].apply(lambda x: 1 if x == 2 else -1)

        # Track dataframes
        dtypes_dict = {
            'Frame ID': np.int64,
            'Vehicle ID': np.int64,
            'Local X': np.float64,
            'Local Y': np.float64,
            'Vehicle Length': np.float64,
            'Vehicle Width': np.float64,
            'Vehicle Velocity X': np.float64,
            'Vehicle Velocity Y': np.float64,
            'Vehicle Acceleration X': np.float64,
            'Vehicle Acceleration Y': np.float64,
            'Front Sight Distance': np.float64,
            'Back Sight Distance': np.float64,
            'Spacing': np.float64,
            'Headway': np.float64,
            'Time to Collision': np.float64,
            'Preceding Velocity X': np.float64,
            'Preceding Vehicle': np.int64,
            'Following Vehicle': np.int64,
            'Left Preceding ID': np.int64,
            'Left Alongside ID': np.int64,
            'Left Following ID': np.int64,
            'Right Preceding ID': np.int64,
            'Right Alongside ID': np.int64,
            'Right Following ID': np.int64,
            'Lane Identification': np.int64,
        }
        df_track = pd.read_csv(os.path.join(file_name, f'{rec}_tracks.csv'),
                               header=0,
                               names=(
                                   'Frame ID',
                                   'Vehicle ID',
                                   'Local X',
                                   'Local Y',
                                   'Vehicle Length',
                                   'Vehicle Width',
                                   'Vehicle Velocity X',
                                   'Vehicle Velocity Y',
                                   'Vehicle Acceleration X',
                                   'Back Sight Distance',
                                   'Spacing',
                                   'Headway',
                                   'Time to Collision',
                                   'Preceding Velocity X',
                                   'Preceding Vehicle',
                                   'Following Vehicle',
                                   'Left Preceding ID',
                                   'Left Alongside ID',
                                   'Left Following ID',
                                   'Right Preceding ID',
                                   'Right Alongside ID',
                                   'Right Following ID',
                                   'Lane Identification'
                               ),
                               dtype=dtypes_dict)
        df_track['Recording ID'] = int(rec)

        # Re-index frames to 0
        df_track['Frame ID'] = df_track['Frame ID'] - 1

        # Compute velocity and acceleration norms
        df_track['Vehicle Velocity'] = np.sqrt(df_track['Vehicle Velocity X']**2 +
                                               df_track['Vehicle Velocity Y']**2)
        df_track['Vehicle Acceleration'] = np.sqrt(df_track['Vehicle Acceleration X'] ** 2 +
                                                   df_track['Vehicle Acceleration Y'] ** 2)
        df_track['Frame ID'] = df_track['Frame ID'] - 1  # re-index frames to 0

        # Merge recordings meta into tracks df on Recording ID key
        rec_cols = ['Recording ID',
                    'Frame Rate',
                    'Location ID',
                    'Month',
                    'Weekday',
                    'Start Time',
                    'Duration',
                    'Upper Lane Markings',
                    'Lower Lane Markings', ]
        merged_rec_df = df_track.join(df_rec[rec_cols].set_index('Recording ID'), on='Recording ID', how='left')

        # Merge tracks meta into tracks df on Vehicle ID key
        merged_tracks_df = merged_rec_df.join(df_track_meta.set_index('Vehicle ID'), on='Vehicle ID', how='left')

        # For each vehicle, get first frame
        vehicle_first_frame = merged_tracks_df.groupby(['Vehicle ID'], sort=False)[['Frame ID']].min()
        vehicle_first_frame.columns = ['First Frame']
        merged_tracks_df = merged_tracks_df.join(vehicle_first_frame, on='Vehicle ID', how='left')

        # Calculate front of vehicle x position
        merged_tracks_df['Local X Front'] = merged_tracks_df['Local X'].where(merged_tracks_df['Driving Direction'] < 0,
                                                                              merged_tracks_df['Local X'] +
                                                                              merged_tracks_df['Vehicle Length'])

        # For each vehicle, get min and max x position
        vehicle_front_min = merged_tracks_df.groupby(['Vehicle ID'], sort=False)[['Local X Front']].min()
        vehicle_front_min.columns = ['Min Local Front']
        vehicle_front_max = merged_tracks_df.groupby(['Vehicle ID'], sort=False)[['Local X Front']].max()
        vehicle_front_max.columns = ['Max Local Front']
        merged_tracks_df = merged_tracks_df.join(vehicle_front_min, on='Vehicle ID', how='left')
        merged_tracks_df = merged_tracks_df.join(vehicle_front_max, on='Vehicle ID', how='left')

        # Save pkl file
        merged_tracks_df.to_pickle(os.path.join(file_name, f'{rec}.pkl'))
        return merged_tracks_df

    def _get_data_frame(self, time_slot, x_max):
        if time_slot in self.cached_data_frames:
            return self.cached_data_frames[time_slot]
        file_name = f'traffic-data/xy-trajectories/highD/'
        if os.path.isfile(os.path.join(file_name, f'{time_slot}.pkl')):
            pkl_file = os.path.join(file_name, f'{time_slot}.pkl')
            print(f'Loading trajectories from {pkl_file}')
            df = pd.read_pickle(pkl_file)
        elif os.path.isfile(os.path.join(file_name, f'{time_slot}_tracks.csv')):
            csv_file = os.path.join(file_name, f'{time_slot}_tracks.csv')
            print(f'Loading trajectories from {csv_file}')
            df = self._read_data_frame(file_name, time_slot)
        else:
            raise FileNotFoundError(f'{file_name + time_slot}.{{pkl,_tracks.csv}} not found.')

        # Remove cars that spontaneously appear on screen
        right_threshold = 150  # any left-to-right cars whose first frame x location is > than this should be removed
        left_threshold = 350  # any right-to-left cars whose first frame x location is < than this should be removed
        df = df[((df['Driving Direction'] > 0) & (df['Min Local Front'] < right_threshold))
                | ((df['Driving Direction'] < 0) & (df['Max Local Front'] > left_threshold))
                | (df['First Frame'] == 0)]  # don't remove any cars that are present in very first frame

        # Calculate X position with offset accounted for so that cars enter and do not spontaneously materialize
        min_of_max = df[(df['Driving Direction'] < 0) & (df['First Frame'] > 0)]['Max Local Front'].min()
        max_of_mins = df[(df['Driving Direction'] > 0) & (df['First Frame'] > 0)]['Min Local Front'].max()
        self.x_offset = {'LOWER': max_of_mins, 'UPPER': self.screen_size[0]/SCALE - min_of_max}
        df['Local Offset X'] = (df['Local X'] - self.x_offset['LOWER']).where(df['Driving Direction'] > 0,
                                                                              df['Local X'] + self.x_offset['UPPER'])
        # Get valid x coordinate rows
        valid_x = (df['Local Offset X'] * SCALE).between(0, x_max)
        # Cache data frame for later retrieval
        self.cached_data_frames[time_slot] = df[valid_x]
        # Restrict data frame to valid x coordinates
        return df[valid_x]

    def _get_first_frame(self, v_id):
        return self.df[self.df['Vehicle ID'] == v_id]['First Frame'].values[0]

    # Need to override _get_neighbours method from traffic_gym parent class to account for two-way traffic
    def _get_neighbours(self, current_lane_idx, d_lane, v):
        # Shallow copy the target lane
        target_lane = self.lane_occupancy[current_lane_idx + d_lane][:]
        # If I find myself in the target list, remove me
        if v in target_lane:
            target_lane.remove(v)
        # Find me in the lane
        my_idx = bisect.bisect(target_lane, v)
        if v._driving_direction > 0:  # check if car is driving left-to-right
            behind = target_lane[my_idx - 1] if my_idx > 0 else None
            ahead = target_lane[my_idx] if my_idx < len(target_lane) else None
        else:
            behind = target_lane[my_idx] if my_idx < len(target_lane) else None
            ahead = target_lane[my_idx - 1] if my_idx > 0 else None
        return behind, ahead

    def reset(self, frame=None, time_slot=None, vehicle_id=None, train_only=False):
        # train_only = True  # uncomment this if doing RL, to set as default behaviour
        if train_only:
            ################################################################################
            # Looping over training split ONLY
            ################################################################################
            if self.train_indx is None:
                train_indx_file = '/home/atcold/Work/GitHub/pytorch-Traffic-Simulator/train_indx.pkl'
                if not os.path.isfile(train_indx_file):
                    pass
                print('Loading training indices')
                with open(train_indx_file, 'rb') as f:
                    self.train_indx = pickle.load(f)
                self.indx_order = list(self.train_indx.keys())
                self.random.shuffle(self.indx_order)
            assert not (frame or time_slot or vehicle_id), 'Already selecting training episode from file.'
            time_slot, vehicle_id = self.train_indx[self.indx_order[self.episode % len(self.indx_order)]]
            self.episode += 1
            ################################################################################

        super().reset(control=(frame is None))
        # print(f'\n > Env on process {os.getpid()} is resetting')
        self._t_slot = self._time_slots[time_slot] if time_slot is not None else self.random.choice(self._time_slots)
        self.df = self._get_data_frame(self._t_slot, self.screen_size[0])
        self.max_frame = max(self.df['Frame ID'])
        if vehicle_id:
            frame = self._get_first_frame(vehicle_id)
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
        self.frame = frame - int(self.delta_t * DT)
        self.vehicles_history = set()

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
                if len(car_df) < 1:
                    continue
                f = self.font[15] if self.display else None
                car = self.EnvCar(car_df, self.look_ahead, self.screen_size[0], f, dt=self.delta_t)
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
                        self.ghost = self.EnvCar(car_df, self.look_ahead, self.screen_size[0], f, dt=self.delta_t)
            self.vehicles_history |= vehicles  # union set operation

        self.lane_occupancy = [[] for _ in range(self.nb_lanes)]
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
            look_sideways = 2 * self.lane_width
            self.render(mode='machine', width_height=(2 * self.look_ahead, 2 * look_sideways), scale=0.25)

        for v in self.vehicles:
            # Generate symbolic state
            lane_idx = v.current_lane
            left_vehicles = self._get_neighbours(lane_idx, -1, v) \
                if 0 < lane_idx < self.nb_lanes-1 or lane_idx == self.nb_lanes-1 and v.front[0] > 18 else None
            mid_vehicles = self._get_neighbours(lane_idx, 0, v)
            right_vehicles = self._get_neighbours(lane_idx, + 1, v) \
                if lane_idx < self.nb_lanes-2 or lane_idx == self.nb_lanes-2 and v.front[0] > 18 else None
            state = left_vehicles, mid_vehicles, right_vehicles

            # Sample an action based on the current state
            action = v.policy() if not v.is_autonomous else policy_action

            # Perform such action
            v.step(action)
            # Uncomment below to display current lane during play maps
            # v._text = v.get_text(f'{v.id}: Lane {v.current_lane}', self.font[15])

            # Store state and action pair
            if (self.store or v.is_controlled) and v.valid:
                v.store('state', state)
                v.store('action', action)

            if v.is_controlled and v.valid:
                v.count_collisions(state)
                if v.collisions_per_frame > 0:
                    self.collision = True

            # # Create set of off track vehicles
            # if v._colour[0] > 128:  # one lane away
            #     if v.id not in self.off_track:
            #         print(f'Adding {v} to off_track set and saving it to disk')
            #         self.off_track.add(v.id)
            #         with open('off_track.pkl', 'wb') as f:
            #             pickle.dump(self.off_track, f)

        # Keep the ghost updated
        if self.store_sim_video:
            if self.ghost and self.ghost.off_screen:
                self.ghost = None
            if self.ghost:
                self.ghost.step(self.ghost.policy())

        self.frame += int(self.delta_t / DT)

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
            if return_:
                return return_

        # return observation, reward, done, info
        return None, None, self.done, None

    def _draw_lanes(self, surface, mode='human', offset=0):
        lanes = self.lanes  # lanes

        if mode == 'human':
            s = surface  # screen
            draw_line = pygame.draw.line  # shortcut
            g = (128, 128, 128)
            sw = self.screen_size[0]  # screen width

            for lane in lanes:
                draw_line(s, g, (0, lane['min']), (sw, lane['min']), 1)
                draw_line(s, g, (0, lane['max']), (sw, lane['max']), 1)
                draw_dashed_line(s, colours['r'], (0, lane['mid']), (sw, lane['mid']))

            top = lanes[0]['min']
            bottom = lanes[-1]['max']
            draw_line(s, (255, 255, 0), (self.look_ahead, top), (self.look_ahead, bottom))
            draw_line(s, (255, 255, 0), (sw - self.look_ahead, top), (sw - self.look_ahead, bottom))
            # pygame.image.save(s, "highD-real.png")

        if mode == 'machine':
            s = surface  # screen
            draw_line = pygame.draw.line  # shortcut
            w = colours['w']  # colour white
            sw = self.screen_size[0]  # screen width
            m = offset

            for lane in lanes:
                draw_line(s, w, (0, lane['min'] + m), (sw + 2 * m, lane['min'] + m), 1)

            top = lanes[0]['min']
            bottom = lanes[-1]['max']
            draw_line(s, (255, 255, 0), (self.look_ahead, top), (self.look_ahead, bottom))
            draw_line(s, (255, 255, 0), (sw - self.look_ahead, top), (sw - self.look_ahead, bottom))

            self._lane_surfaces[mode] = surface.copy()
            # pygame.image.save(surface, "highD-machine.png")
