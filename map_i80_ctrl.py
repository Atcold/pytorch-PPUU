import bisect
import os

from map_i80 import I80, I80Car
from traffic_gym import Car, Simulator
from traffic_gym_v2 import PatchedCar


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

    def __init__(self, lanes, free_lanes, dt, car_id, look_ahead, screen_w, font, is_controlled=False, bot_speed=0):
        super().__init__(lanes, free_lanes, dt, car_id, look_ahead, screen_w, font,
                         policy_type=None, policy_network=None)
        self.buffer_size = 0
        self.is_controlled = is_controlled
        self._speed = bot_speed
        self.lanes = lanes
        self.LANE_W = 24  # pixels / 3.7 m, lane width
        self.SCALE = self.LANE_W / 3.7  # pixels per metre
        self.arrived_to_dst = False
        self.off_screen = False
        self.frames = list()

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
        #                 print(f'Collision {self.collisions_per_frame}/6, behind, vehicle {behind.id}')
        #         if ahead:
        #             d = ahead - self
        #             if d[0] < -alpha and abs(d[1]) + alpha < (self._width + ahead._width) / 2:
        #                 self.collisions_per_frame += 1
        #                 print(f'Collision {self.collisions_per_frame}/6, ahead, vehicle {ahead.id}')

        beta = 0.99
        if self._states_image and self._states_image[-1][2] > beta:
            self.collisions_per_frame += 1
            print(f'Collision registered for vehicle {self}')
            print(f'Accident! Check vehicle {self}. Proximity of {self._states_image[-1][2]}.')


class SimI80(Simulator):
    EnvCar = Sim80Car

    # Import map from Simulator
    _draw_lanes = I80._draw_lanes

    def __init__(self, **kwargs):
        kwargs['nb_lanes'] = 6
        kwargs['delta_t'] = 1/10
        super().__init__(**kwargs)
        self.nb_lanes = 7
        self.screen_size = (85 * self.LANE_W, self.nb_lanes * self.LANE_W + 5 * self.LANE_W)
        self.counter = 0
        self.BotCarSpeed = 117
        self.input_images = None
        self.input_states = None
        self.controlled_car = dict(locked=False)
        # Conversion LANE_W from real world to pixels: US highway lane width is 3.7 metres, here 50 pixels
        self.LANE_W = 24  # pixels / 3.7 m, lane width
        self.SCALE = self.LANE_W / 3.7  # pixels per metre
        self.screen_size = (160 * self.LANE_W, self.nb_lanes * self.LANE_W + self.offset + self.LANE_W // 2)
        self.policy = 'MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-' \
             'lambdal=0.0-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=False-learnedcost=True-lambdatl=1.0' \
             '-seed=3-novaluestep85000.model'
        self.dump_folder = os.path.join('simulator-dumps', 'train-of-cars', self.policy)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        observation = None
        while observation is None:
            observation, reward, done, _ = self.step()
        return observation

    def step(self, policy_action=None):
        assert not self.done, 'Trying to step on an exhausted environment!'
        f = self.font[20] if self.display else None
        if self.frame % 5 == 0:
            # Add 3 train cars in a row
            if self.counter < 4:
                car = self.EnvCar(self.lanes, [2], self.delta_t, self.next_car_id,
                                  self.look_ahead, self.screen_size[0], f,
                                  is_controlled=False,
                                  bot_speed=self.BotCarSpeed)
                car._speed = self.BotCarSpeed
                self.next_car_id += 1
                self.vehicles.append(car)
                self.counter += 1
                if self.next_car_id <= 3:
                    for i in [0, 1, 3, 4, 5]:
                        car = self.EnvCar(self.lanes, [i], self.delta_t, self.next_car_id,
                                          self.look_ahead, self.screen_size[0], f,
                                          is_controlled=False,
                                          bot_speed=self.BotCarSpeed)
                        self.vehicles.append(car)
                elif self.next_car_id == 4:
                    car = self.EnvCar(self.lanes, [4], self.delta_t, self.next_car_id,
                                      self.look_ahead, self.screen_size[0], f,
                                      is_controlled=False,
                                      bot_speed=self.BotCarSpeed)
                    self.vehicles.append(car)
                elif self.next_car_id == 5:
                    controlled_car = self.EnvCar(self.lanes, [3], self.delta_t, self.next_car_id,
                                                 self.look_ahead, self.screen_size[0], f,
                                                 is_controlled=True,
                                                 bot_speed=self.BotCarSpeed)
                    self.controlled_car['locked'] = controlled_car
                    self.next_car_id += 1
                    self.vehicles.append(controlled_car)
                    car = self.EnvCar(self.lanes, [4], self.delta_t, self.next_car_id,
                                      self.look_ahead, self.screen_size[0], f,
                                      is_controlled=False,
                                      bot_speed=self.BotCarSpeed)
                    self.vehicles.append(car)
                elif self.next_car_id > 6:
                    for i in [0, 1, 3, 4, 5]:
                        car = self.EnvCar(self.lanes, [i], self.delta_t, self.next_car_id,
                                          self.look_ahead, self.screen_size[0], f,
                                          is_controlled=False,
                                          bot_speed=self.BotCarSpeed)
                        self.vehicles.append(car)
            elif self.frame < 55:
                car = self.EnvCar(self.lanes, [0], self.delta_t, self.next_car_id,
                                  self.look_ahead, self.screen_size[0], f,
                                  is_controlled=False,
                                  bot_speed=self.BotCarSpeed)
                self.vehicles.append(car)
                car = self.EnvCar(self.lanes, [4], self.delta_t, self.next_car_id,
                                  self.look_ahead, self.screen_size[0], f,
                                  is_controlled=False,
                                  bot_speed=self.BotCarSpeed)
                self.vehicles.append(car)
        # Create space after first 3 train cars
        if self.frame == 30:
            self.counter = 0

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
            if v.is_controlled and self.controlled_car['locked'] and policy_action is not None:
                action = policy_action

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
                print(f' Current y position: {self.controlled_car["locked"]._position[1]}; Action: {action}')

            if v.is_controlled and self.controlled_car['locked']:
                v.count_collisions(state)
                if v.collisions_per_frame > 0:
                    self.collision = True
                    self.done = True  # stop when an accident occurs

        self.frame += 1
        if self.controlled_car and self.controlled_car['locked']:
            return_ = self.controlled_car['locked'].get_last(n=self.nb_states, done=self.done)
            if return_:
                return return_
        return None, None, self.done, None
