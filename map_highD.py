import pdb
import bisect
import os
import pickle

import numpy as np
import pandas as pd
import pygame
import torch

from custom_graphics import draw_highD_rect, draw_dashed_line
from traffic_gym import Simulator, Car, colours

# TODO: Review these global variables
# SCALE_X = 1
# SCALE_Y = 2
# VERTICAL_OFFSET = 25
# bounding_box /= 0.10106
# bounding_box /= 4
# X_OFFSET = 50  # horizontal offset - arbitrary number of pixels to have at beginning of recording
# Y_OFFSET = 25  # arbitrary amount of pixels to have as buffer on either side of the highway
SCALE = 5
Y_OFFSET = 50
X_OFFSET = 60
MAX_SPEED = 130
DT = 1 / 25  # frame rate for highD dataset (comes from recording_meta csv files)


# Recording meta data
def read_recoding_meta(filename, reload=False):
    # Check if concatenated file already exists:
    if os.path.isfile(os.path.join(filename, 'all_recordingMeta.csv')) and not reload:
        df_recs = pd.read_csv(os.path.join(filename, 'all_recordingMeta.csv'))
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
    df_recs.to_csv(os.path.join(filename, 'all_recordingMeta.csv'), index=False)
    return df_recs


class HighDCar(Car):
    # Global constants
    max_a = 40  # TODO: check where this number is used and if it needs to change
    max_b = 0.01  # TODO: check where this number is used and if it needs to change

    def __init__(self, df, look_ahead, screen_w, font=None, kernel=0, dt=DT):
        # k = kernel  # running window size
        self._length = df.at[df.index[0], 'Vehicle Length'] * SCALE
        self._width = df.at[df.index[0], 'Vehicle Width'] * SCALE
        self.id = df.at[df.index[0], 'Vehicle ID']  # extract scalar <'Vehicle ID'> <at> <index[0]>

        # x = df['Local X'].rolling(window=k).mean().shift(1 - k).values
        # y = df['Local Y'].rolling(window=k).mean().shift(1 - k).values
        x = df['Local X'].values * SCALE + X_OFFSET
        y = df['Local Y'].values * SCALE + Y_OFFSET
        if dt > DT:
            s = int(dt / DT)
            end = len(x) - len(x) % s
            x = x[:end].reshape(-1, s).mean(axis=1)
            y = y[:end].reshape(-1, s).mean(axis=1)
        self._max_t = len(x) - np.count_nonzero(np.isnan(x)) - 2  # 2 for computing the acceleration
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
        self.collisions_per_frame = 0

    @property
    def is_autonomous(self):
        return False

    def _get(self, what, k):
        direction_vector = self._trajectory[k + 1] - self._trajectory[k]
        norm = np.linalg.norm(direction_vector)
        if what == 'direction':
            if norm < 1e-6:
                return self._direction  # if static returns previous direction
            return direction_vector / norm
        if what == 'speed':
            return norm / self._dt
        if what == 'init_direction':  # valid direction can be computed when speed is non-zero
            t = 1  # check if the car is in motion the next step
            while self._df.at[self._df.index[t], 'Vehicle Velocity'] < 5 and t < self._max_t:
                t += 1
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
        current_y = self._trajectory[self._frame][1]
        if current_y > self._upper_lane_markings[-1]:  # Check if in lower lanes
            res = list(map(lambda i: i > current_y, self._lower_lane_markings)).index(True)
            return int(res + (len(self._upper_lane_markings) / 2))
        # Else, in upper lanes
        res = list(map(lambda i: i > current_y, self._upper_lane_markings)).index(True)
        return int(res + 1)

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
            print(f'Collision registered for vehicle {self}')
            print(f'Accident! Check vehicle {self}. Proximity of {self._states_image[-1][2]}.')

    def draw(self, surface, mode='human', offset=0):
        """
        Draw current car on screen with a specific colour
        :param surface: PyGame ``Surface`` where to draw
        :param mode: human or machine
        :param offset: for representation cropping
        """
        x, y = self._position + offset
        rectangle = (x, y, self._length, self._width)

        d = self._direction

        if mode == 'human':
            if self.is_controlled:
                pygame.draw.rect(surface, (0, 255, 0),
                                 (int(x - 10), int(y - 15), self._length + 10 + 10, 30), 2)

            # # Highlight colliding vehicle / debugging purpose
            # if self.collisions_per_frame > 0:
            #     larger_rectangle = (*((x, y) - self._direction * 10), self._length + 10 + 10, self._width + 10 + 10,)
            #     draw_rect(surface, colours['g'], larger_rectangle, d, 2)
            #     # # Remove collision, if reading it from file
            #     # self.collisions_per_frame = 0

            # # Pick one out
            # if self.id == 738: self._colour = colours['r']

            # # Green / red -> left-to-right / right-to-left
            # if d[0] > 0: self._colour = (0, 255, 0)  # green: vehicles moving to the right
            # if d[0] < 0: self._colour = (255, 0, 0)  # red: vehicles moving to the left

            _r = draw_highD_rect(surface, self._colour, rectangle, d)

            # Drawing vehicle number
            if x < self.front[0]:
                self._text[1].left = x + self._length / 4
            else:
                self._text[1].right = x - self._length / 4
            self._text[1].top = y + self._width / 4
            surface.blit(self._text[0], self._text[1])

            if self._braked:
                self._colour = colours['g']
            return _r
        if mode == 'machine':
            return draw_highD_rect(surface, colours['g'], rectangle, d)
        if mode == 'ego-car':
            return draw_highD_rect(surface, colours['b'], rectangle, d)
        if mode == 'ghost':
            return draw_highD_rect(surface, colours['y'], rectangle, d)


class HighD(Simulator):
    # Environment's car class
    EnvCar = HighDCar

    # Global constants
    DUMP_NAME = 'data_highD_v0'

    def __init__(self, **kwargs):
        self.rec_meta = read_recoding_meta('traffic-data/xy-trajectories/highD/', reload=True)
        kwargs['nb_lanes'] = self.rec_meta[self.rec_meta['Recording ID']==int(kwargs['rec'])]["Number Lanes"].values[0]
        self.recording = kwargs['rec']
        del kwargs['rec']
        delta_t = kwargs['delta_t']
        assert delta_t >= DT, f'Minimum delta t is 1 / 25 s > {delta_t:.2f}s you tried to set'
        assert (delta_t / DT).is_integer(), f'dt: {delta_t:.2f}s must be a multiple of 1 / 25 s'

        super().__init__(**kwargs)

        self.screen_size = (512, 256)
        photo = pygame.image.load(f'HighD/{self.recording}_highway.png')
        photo = pygame.transform.scale(photo, self.screen_size)
        # self.photos = (photo,)
        # self.photos_rect = (self.photos[0].get_rect().move([0, 0]),)
        print(photo.get_size()[0], photo.get_size()[1])
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
        self.nb_lanes = kwargs['nb_lanes']
        self.lanes = self.build_lanes(kwargs['nb_lanes'])
        self.smoothing_window = 15
        self.max_frame = -1
        pth = 'traffic-data/state-action-cost/data_highD_v0/data_stats.pth'
        self.data_stats = torch.load(pth) if self.normalise_state or self.normalise_action else None
        self.cached_data_frames = dict()
        self.episode = 0
        self.train_indx = None
        self.indx_order = None

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
                                   'Vehicle Velocity',
                                   'Vehicle Velocity Y',
                                   'Vehicle Acceleration',
                                   'Vehicle Acceleration Y',
                                   'Front Sight Distance',
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
        df_track["Recording ID"] = int(rec)
        df_track["Frame ID"] = df_track["Frame ID"] - 1  # re-index frames to 0

        # df_track["Vehicle Velocity"] = np.sign(df_track["Vehicle Velocity X"]) * \
        #                                np.sqrt(df_track["Vehicle Velocity X"]**2 +
        #                                        df_track["Vehicle Velocity Y"]**2)
        # df_track["Vehicle Acceleration"] = np.sqrt(df_track["Vehicle Acceleration X"] ** 2 +
        #                                            df_track["Vehicle Acceleration Y"] ** 2)

        # Merge recordings meta into tracks df on Recording ID key
        rec_cols = ['Recording ID',
                    'Frame Rate',
                    'Location ID',
                    'Month',
                    'Weekday',
                    'Start Time',
                    'Duration',
                    'Upper Lane Markings',
                    'Lower Lane Markings',]
        merged_rec_df = df_track.join(df_rec[rec_cols].set_index('Recording ID'), on='Recording ID', how='left')

        # Merge tracks meta into tracks df on Vehicle ID key
        merged_tracks_df = merged_rec_df.join(df_track_meta.set_index('Vehicle ID'), on='Vehicle ID', how='left')

        # Save pkl file
        merged_tracks_df.to_pickle(os.path.join(file_name, f'{rec}.pkl'))
        return merged_tracks_df

    def _get_data_frame(self, time_slot, y_max):
        # if time_slot in self.cached_data_frames:
        #     return self.cached_data_frames[time_slot]
        # file_name = f'traffic-data/xy-trajectories/highD/'
        # df_dict = dict()
        # # Determine which recordings to use (recordings are numbered "01" to "60")
        # if "-" in time_slot:  # indicates that range of multiple recordings was given, e.g. "03-11"
        #     start_rec_num = time_slot.split('-')[0]
        #     end_rec_num = time_slot.split('-')[1]
        #     recs_list = [f'{i:02d}' for i in range(int(start_rec_num), int(end_rec_num) + 1)]
        # else:  # else, only one recording was given, e.g. "15"
        #     recs_list = [time_slot]
        # # For each recording read csv / pkl file and add it to dictionary of dataframes
        # for rec in recs_list:
        #     if os.path.isfile(os.path.join(file_name, f'{rec}.pkl')):
        #         pkl_file = os.path.join(file_name, f'{rec}.pkl')
        #         print(f'Loading trajectories from {file_name}')
        #         df_dict["rec"] = pd.read_pickle(pkl_file)
        #     elif os.path.isfile(os.path.join(file_name, f'{rec}_tracks.csv')):
        #         csv_file = os.path.join(file_name, f'{rec}_tracks.csv')
        #         print(f'Loading trajectories from {csv_file}')
        #         df_dict["rec"] = self._read_data_frame(file_name, rec)
        #     else:
        #         raise FileNotFoundError(f'{file_name + time_slot}.{{pkl,_tracks.csv}} not found.')
        # # Concatenate the dictionary of dataframes into single dataframe
        # df = pd.concat([df_dict[rec] for rec in df_dict], ignore_index=True)
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

        # Get valid x coordinate rows
        valid_y = (df['Local Y']).between(0, y_max)

        # Cache data frame for later retrieval
        self.cached_data_frames[time_slot] = df[valid_y]

        # Restrict data frame to valid x coordinates
        return df[valid_y]

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
                if len(car_df) < self.smoothing_window + 1:
                    continue
                f = self.font[15] if self.display else None
                car = self.EnvCar(car_df, self.look_ahead, self.screen_size[0], f, self.smoothing_window,
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
                if 0 < lane_idx < 6 or lane_idx == 6 and v.front[0] > 18 else None
            mid_vehicles = self._get_neighbours(lane_idx, 0, v)
            right_vehicles = self._get_neighbours(lane_idx, + 1, v) \
                if lane_idx < 5 or lane_idx == 5 and v.front[0] > 18 else None
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
                if v.collisions_per_frame > 0:
                    self.collision = True

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
            w = colours['w']  # colour white
            g = (128, 128, 128)
            sw = self.screen_size[0]  # screen width

            for lane in lanes:
                draw_line(s, g, (0, lane['min']), (sw, lane['min']), 1)
                draw_line(s, g, (0, lane['max']), (sw, lane['max']), 1)
                draw_dashed_line(s, colours['r'], (0, lane['mid']), (sw, lane['mid']))

            # draw_line(s, w, (0, lanes[0]['min']), (sw, lanes[0]['min']), 3)
            bottom = lanes[-1]['max']
            # draw_line(s, w, (0, bottom), (sw, bottom), 3)
            # draw_line(s, w, (0, bottom + 29), (18, bottom + 29 - slope * 18), 3)
            # draw_line(s, g, (18, bottom + 13), (31, bottom), 1)
            # # draw_line(s, g, (0, bottom + 42), (60 * LANE_W, bottom + 42 - slope * 60 * LANE_W), 1)
            # draw_line(s, w, (0, bottom + 53), (60, bottom + 53 - slope * 60), 3)
            # draw_line(s, w, (60, bottom + 3), (sw, bottom + 2), 3)

            # look_ahead = MAX_SPEED * 1000 / 3600 * self.SCALE
            # o = self.offset
            look_ahead = MAX_SPEED * 1000 / 3600 * SCALE
            o = X_OFFSET
            draw_line(s, (255, 255, 0), (look_ahead, o), (look_ahead, 9.4))
            draw_line(s, (255, 255, 0), (sw - 1.75 * look_ahead, o), (sw - 1.75 * look_ahead, bottom))
            draw_line(s, (255, 255, 0), (sw - 0.75 * look_ahead, o), (sw - 0.75 * look_ahead, bottom), 5)
            # pygame.image.save(s, "i80-real.png")

        if mode == 'machine':
            s = surface  # screen
            draw_line = pygame.draw.line  # shortcut
            w = colours['r']  # colour white
            sw = self.screen_size[0]  # screen width
            m = offset

            for lane in lanes:
                draw_line(s, w, (0, lane['min'] + m), (sw + 2 * m, lane['min'] + m), 1)

            # bottom = lanes[-1]['max'] + m
            # draw_line(s, w, (0, bottom), (m + 18, bottom), 1)
            # draw_line(s, w, (m, bottom + 29), (m + 18, bottom + 29 - slope * 18), 1)
            # draw_line(s, w, (m + 18, bottom + 13), (m + 31, bottom), 1)
            # draw_line(s, w, (m, bottom + 53), (m + 60, bottom + 53 - slope * 60), 1)
            # draw_line(s, w, (m + 60, bottom + 3), (2 * m + sw, bottom), 1)

            self._lane_surfaces[mode] = surface.copy()
            # pygame.image.save(surface, "i80-machine.png")
