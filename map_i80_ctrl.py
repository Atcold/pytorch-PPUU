from os import path
from map_i80 import I80, I80Car
from traffic_gym_v2 import PatchedCar, MergingMap
from traffic_gym import Car, Simulator
import bisect
import pygame
import torch
import ipdb


class ControlledI80Car(I80Car):
    # Import get_lane_set from PatchedCar
    get_lane_set = PatchedCar.get_lane_set

    def __init__(self, df, y_offset, look_ahead, screen_w, font=None, kernel=0, dt=1 / 10):
        super().__init__(df, y_offset, look_ahead, screen_w, font, kernel, dt)
        self.is_controlled = False
        self.buffer_size = 0
        self.lanes = None
        self.arrived_to_dst = False  # arrived to destination
        self.frames = list()

    @property
    def current_lane(self):
        # If following the I-80 trajectories
        if not self.is_controlled or len(self._states_image) < self.buffer_size:
            return super().current_lane

        # Otherwise fetch x location
        x = self._position[0]
        if x > self.screen_w - 1.75 * self.look_ahead:
            self.off_screen = True
            self.arrived_to_dst = True

        # Fetch the y location
        y = self._position[1]

        # If way too up
        if y < self.lanes[0]['min']:
            self.off_screen = True
            self.arrived_to_dst = False
            return 0

        # Maybe within a sensible range?
        for lane_idx, lane in enumerate(self.lanes):
            if lane['min'] <= y <= lane['max']:
                return lane_idx

        # Or maybe on the ramp
        bottom = self.lanes[-1]['max']
        if y <= bottom + 53 - x * 0.035:
            return 6

        # Actually, way too low
        self.off_screen = True
        self.arrived_to_dst = False
        return 6

    @property
    def is_autonomous(self):
        return self.is_controlled and len(self._states_image) > self.buffer_size


class ControlledI80(I80):
    # Environment's car class
    EnvCar = ControlledI80Car

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        observation = None
        first_frame = self.frame
        while observation is None:
            observation, reward, done, _ = self.step()
        return observation, first_frame


class Sim80Car(Car):
    current_lane = ControlledI80Car.current_lane

    def __init__(self, lanes, free_lanes, dt, car_id, look_ahead, screen_w, font, policy_type,
                 is_controlled=False, bot_speed=0):
        super().__init__(lanes, free_lanes, dt, car_id, look_ahead, screen_w, font, policy_type, policy_network=None)
        self.buffer_size = 0
        self.is_controlled = is_controlled
        self._speed = bot_speed
        self.lanes = lanes
        self.LANE_W = 24  # pixels / 3.7 m, lane width
        self.SCALE = self.LANE_W / 3.7  # pixels per metre

    def count_collisions(self, state):
        self.collisions_per_frame = 0
        alpha = 1 * self.SCALE  # 1 m overlap collision
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


class SimI80(MergingMap):
    EnvCar = Sim80Car

    # ControlledCar = ControlledI80Car

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.screen_size = (170 * self.LANE_W, self.nb_lanes * self.LANE_W + 5 * self.LANE_W)
        # self.screen_size = (85 * self.LANE_W, self.nb_lanes * self.LANE_W + 5 * self.LANE_W)
        self.counter = 0
        self.BotCarSpeed = 117
        self.input_images = None
        self.input_states = None
        if self.display:  # if display is required
            self.screen = pygame.display.set_mode(self.screen_size)  # set screen size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_sequences = []
        self.data_path = 'traffic-data/state-action-cost/data_i80_v0'
        # self.model_dir = '/Users/yairschiff/nvidia-collab/yairschiff/pytorch-PPUU/models_learned_cost'
        self.model_dir = 'models_learned_cost'
        self.mfile = 'model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-' + \
                     'nz=32-beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model'
        self.policy_model = 'MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30' + \
                            '-ureg=0.05-lambdal=0.0-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=False' + \
                            '-learnedcost=True-lambdatl=1.0-seed=2-novaluestep100000.model'
        self.forward_model, self.data_stats = self.load_models()
        self.forward_model.to(self.device)
        self.npred = 20
        self.LANE_W = 24
        self.nb_states = 20
        self.dump_folder = path.join('train-of-cars', self.policy_model)

    def load_models(self):
        stats = torch.load(path.join(self.data_path, 'data_stats.pth'))  # , map_location=self.device)
        forward_model = torch.load(path.join(self.model_dir, self.mfile))  # , map_location=self.device)
        if type(forward_model) is dict: forward_model = forward_model['model']
        model_path = path.join(self.model_dir, f'policy_networks/{self.policy_model}')
        policy_network_mpur = torch.load(model_path)['model']  # , map_location=self.device)['model']
        policy_network_mpur.stats = stats
        forward_model.policy_net = policy_network_mpur.policy_net
        forward_model.policy_net.stats = stats
        forward_model.policy_net.actor_critic = False
        return forward_model, stats

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.controlled_car = {
            'locked': None,
        }

    def step(self, policy_action=None):
        assert not self.done, 'Trying to step on an exhausted environment!'
        if self.frame % 5 == 0:
            # Add 3 train cars in a row
            if self.counter < 4:
                car = self.EnvCar(self.lanes, [2], self.delta_t, self.next_car_id,
                                  self.look_ahead, self.screen_size[0], self.font[20],
                                  policy_type='straight',
                                  is_controlled=False,
                                  bot_speed=self.BotCarSpeed)
                car._speed = self.BotCarSpeed
                self.next_car_id += 1
                self.vehicles.append(car)
                self.counter += 1
                if self.next_car_id == 3:
                    car = self.EnvCar(self.lanes, [3], self.delta_t, self.next_car_id,
                                      self.look_ahead, self.screen_size[0], self.font[20],
                                      policy_type='straight',
                                      is_controlled=False,
                                      bot_speed=self.BotCarSpeed)
                    car._speed = self.BotCarSpeed
                    self.next_car_id += 1
                    self.vehicles.append(car)
                    self.counter += 1
                if self.next_car_id == 6:
                    controlled_car = self.EnvCar(self.lanes, [3], self.delta_t, self.next_car_id,
                                                 self.look_ahead, self.screen_size[0], self.font[20],
                                                 policy_type='controlled',
                                                 is_controlled=True,
                                                 bot_speed=self.BotCarSpeed)
                    self.next_car_id += 1
                    self.vehicles.append(controlled_car)
            # Create space after first 3 train cars
        if self.frame == 25:
            self.counter = 0

        if self.frame == 77:
            self.controlled_car['locked'] = self.vehicles[5]

        if self.show_frame_count:
            print(f'\r[t={self.frame}]', end='')

        for v in self.vehicles:
            lanes_occupied = v.get_lane_set(self.lanes)
            # Check for any passing and update lane_occupancy
            for l in range(self.nb_lanes):
                if l in lanes_occupied and v not in self.lane_occupancy[l]:
                    # Enter lane
                    bisect.insort(self.lane_occupancy[l], v)
                elif l not in lanes_occupied and v in self.lane_occupancy[l]:
                    # Leave lane
                    self.lane_occupancy[l].remove(v)
            # Remove from the environment cars outside the screen
            if v.back[0] > self.screen_size[0]:
                for l in lanes_occupied: self.lane_occupancy[l].remove(v)
                self.vehicles.remove(v)

            # Bot cars should not apply any acceleration
            action = (0, 0)
            # Controlled car should execute action based on policy
            if v.is_controlled and self.controlled_car['locked']:
                transpose = list(zip(*v._states_image))
                state_images = transpose[0]
                state_images = torch.stack(state_images).permute(0, 3, 1, 2)[-self.nb_states:]

                zip_ = list(zip(*v._states))  # n × (obs, mask, cost) -> (n × obs, n × mask, n × cost)
                states = torch.stack(zip_[0])[:, 0][
                         -self.nb_states:]  # select the ego-state (of 1 + 6 states we keep track)
                # if self.norm_state is not False:  # normalise the states, if requested
                states = states.sub(self.data_stats['s_mean']).div(self.data_stats['s_std'])  # N(0, 1) range
                state_images = state_images.float().div(255)  # [0, 1] range

                # Get action based on policy
                self.forward_model.reset_action_buffer(self.npred)
                self.action_sequences.append([])
                # target = 96.0
                # pain_factor = 7
                # distance_to_target = v._position[1] - target
                # exaggerated_target_y = v._position[1] - \
                #                        distance_to_target*pain_factor * (1 if distance_to_target > 0 else -1)
                # target_y = torch.tensor(exaggerated_target_y).to(self.device)
                # print(f'Target y: {exaggerated_target_y}')
                # if abs(distance_to_target) > 10:
                #     target_y = torch.tensor(5.0).to(self.device)
                # else:
                target_y = torch.tensor(96.0).to(self.device)
                a, _, _, _ = self.forward_model.policy_net(state_images, states, sample=True,
                                                           normalize_inputs=True, normalize_outputs=True,
                                                           controls=dict(target_lanes=target_y))
                action = a.cpu().view(1, 2).numpy()
                action = (action[0][0], action[0][1])
                self.action_sequences[-1].append(action)
                print(f'Frame {self.frame}: Action - {action}; Y-pos - {v._position[1]}')

            # Perform such action
            v.step(action)

            if v.is_controlled:
                self.render(mode='machine', width_height=(2 * self.look_ahead, 2 * self.look_sideways), scale=0.25)
                # Generate symbolic state
                lane_idx = v.current_lane
                left_vehicles = self._get_neighbours(lane_idx, -1, v) \
                    if 0 < lane_idx < 6 or lane_idx == 6 and v.front[0] > 18 * self.LANE_W else None
                mid_vehicles = self._get_neighbours(lane_idx, 0, v)
                right_vehicles = self._get_neighbours(lane_idx, + 1, v) \
                    if lane_idx < 5 or lane_idx == 5 and v.front[0] > 18 * self.LANE_W else None
                state = left_vehicles, mid_vehicles, right_vehicles
                v.store('state', state)
                v.store('action', action)

            if v.is_controlled and self.controlled_car['locked']:
                v.count_collisions(state)
                if v.collisions_per_frame > 0:
                    self.collision = True
                    self.done = True  # stop when an accident occurs

        self.frame += 1
        return None, None, self.done, None
