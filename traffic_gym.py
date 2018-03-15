import bisect

import pygame, pdb, torch
import math
import random
import numpy as np
import scipy.misc
import sys
from custom_graphics import draw_dashed_line, draw_text, draw_rect
from gym import core
import os
from scipy.misc import imsave

# Conversion LANE_W from real world to pixels
# A US highway lane width is 3.7 metres, here 50 pixels
LANE_W = 20  # pixels / 3.7 m, lane width
SCALE = LANE_W / 3.7  # pixels per metre

colours = {
    'w': (255, 255, 255),
    'k': (000, 000, 000),
    'r': (255, 000, 000),
    'g': (000, 255, 000),
    'm': (255, 000, 255),
    'b': (000, 000, 255),
    'c': (000, 255, 255),
    'y': (255, 255, 000),
}

# Car coordinate system, origin under the centre of the read axis
#
#      ^ y                       (x, y, x., y.)
#      |
#   +--=-------=--+
#   |  | z        |
# -----o-------------->
#   |  |          |    x
#   +--=-------=--+
#      |
#
# Will approximate this as having the rear axis on the back of the car!
#
# Car sizes:
# type    | width [m] | length [m]
# ---------------------------------
# Sedan   |    1.8    |    4.8
# SUV     |    2.0    |    5.3
# Compact |    1.7    |    4.5

MAX_SPEED = 130  # km/h


class Car:

    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W

    def __init__(self, lanes, free_lanes, dt, car_id):
        """
        Initialise a sedan on a random lane
        :param lanes: tuple of lanes, with ``min`` and ``max`` y coordinates
        :param dt: temporal updating interval
        """
        self._length = round(4.8 * self.SCALE)
        self._width = round(1.8 * self.SCALE)
        self._direction = np.array((1, 0), np.float)
        self.id = car_id
        lane = random.choice(tuple(free_lanes))
        self._position = np.array((
            -self._length,
            lanes[lane]['mid']
        ), np.float)
        self._target_speed = max(30, (MAX_SPEED - random.randrange(0, 30) - 10 * lane)) * 1000 / 3600 * self.SCALE  # m / s
        self._speed = self._target_speed
        self._dt = dt
        self._colour = colours['c']
        self._braked = False
        self._passing = False
        self._target_lane = self._position[1]
        self._target_lane_ = self._target_lane
        self.crashed = False
        self._error = 0
        self._states = list()
        self._states_image = list()
        self._actions = list()
        self._safe_factor = random.gauss(1, .2)  # 0.9 Germany, 2 safe
        self.pid_k1 = 0.01 + np.random.normal(0, 0.005)
        self.pid_k2 = 0.3 + np.random.normal(0, 0.02)

    def get_state(self):
        state = torch.zeros(4)
        state[0] = self._position[0]  # x
        state[1] = self._position[1]  # y
        state[2] = self._direction[0] * self._speed  # dx
        state[3] = self._direction[1] * self._speed  # dy
        return state

    def _get_obs(self, left_vehicles, mid_vehicles, right_vehicles):
        n_cars = 1 + 6  # this car + 6 neighbors
        obs = torch.zeros(n_cars, 2, 2)
        mask = torch.zeros(n_cars)
        obs = obs.view(n_cars, 4)

        obs[0].copy_(self.get_state())

        if left_vehicles:
            if left_vehicles[0] is not None:
                obs[1].copy_(left_vehicles[0].get_state())
                mask[1] = 1
            else:
                # for bag-of-cars this will be ignored by the mask,
                # but fill in with a similar value to not mess up batch norm
                obs[1].copy_(self.get_state())

            if left_vehicles[1] is not None:
                obs[2].copy_(left_vehicles[1].get_state())
                mask[2] = 1
            else:
                obs[2].copy_(self.get_state())
        else:
            obs[1].copy_(self.get_state())
            obs[2].copy_(self.get_state())

        if mid_vehicles[0] is not None:
            obs[3].copy_(mid_vehicles[0].get_state())
            mask[3] = 1
        else:
            obs[3].copy_(self.get_state())

        if mid_vehicles[1] is not None:
            obs[4].copy_(mid_vehicles[1].get_state())
            mask[4] = 1
        else:
            obs[4].copy_(self.get_state())

        if right_vehicles:
            if right_vehicles[0] is not None:
                obs[5].copy_(right_vehicles[0].get_state())
                mask[5] = 1
            else:
                obs[5].copy_(self.get_state())

            if right_vehicles[1] is not None:
                obs[6].copy_(right_vehicles[1].get_state())
                mask[6] = 1
            else:
                obs[6].copy_(self.get_state())
        else:
            obs[5].copy_(self.get_state())
            obs[6].copy_(self.get_state())

        return obs, mask

    def draw(self, surface, c=False, mode='human', offset=0):
        """
        Draw current car on screen with a specific colour
        :param surface: PyGame ``Surface`` where to draw
        :param c: default colour
        :param mode: human or machine
        :param offset: for representation cropping
        :param scale: draw with rescaled coordinates
        """
        x, y = self._position + offset
        rectangle = (int(x), int(y), self._length, self._width)
        if mode == 'human':
            if c:
                pygame.draw.rect(surface, (0, 255, 0),
                                 (int(x - 15), int(y - 15), self._length + 20, self._width + 20), 2)
            draw_rect(surface, self._colour, rectangle, self._direction, 3)
            draw_text(surface, str(self.id), (x, y - self._width // 2), 20, colours['b'])
            if self._braked: self._colour = colours['g']
        if mode == 'machine':
            draw_rect(surface, colours['g'], rectangle, self._direction)

    def step(self, action):  # takes also the parameter action = state temporal derivative
        """
        Update current position, given current velocity and acceleration
        """
        # Vehicle state definition
        vehicle_state = np.array((*self._position, *self._direction, self._speed))
        # State integration
        d_position_dt = self._speed * self._direction
        vehicle_state[:2] += d_position_dt * self._dt
        vehicle_state[2:] += action * self._dt

        # Split individual components (and normalise direction)
        self._position = vehicle_state[0:2]
        self._direction = vehicle_state[2:4] / np.linalg.norm(vehicle_state[2:4])
        self._speed = vehicle_state[4]

        # Deal with latent variable and visual indicator
        if self._passing and abs(self._error) < 0.5:
            self._passing = False
            self._colour = colours['c']

    def get_lane_set(self, lanes):
        """
        Returns the set of lanes currently occupied
        :param lanes: tuple of lanes, with ``min`` and ``max`` y coordinates
        :return: busy lanes set
        """
        busy_lanes = set()
        y = self._position[1]
        half_w = self._width // 2
        for lane_idx, lane in enumerate(lanes):
            if lane['min'] <= y - half_w <= lane['max'] or lane['min'] <= y + half_w <= lane['max']:
                busy_lanes.add(lane_idx)
        return busy_lanes

    @property
    def safe_distance(self):
        return self._speed * self._safe_factor

    @property
    def front(self):
        return int(self._position[0] + self._length)

    @property
    def back(self):
        return int(self._position[0])

    def _brake(self, fraction):
        if self._passing: return 0
        # Maximum braking acceleration, eq. (1) from
        # http://www.tandfonline.com/doi/pdf/10.1080/16484142.2007.9638118
        g, mu = 9.81, 0.9  # gravity and friction coefficient
        acceleration = -fraction * g * mu * self.SCALE
        self._colour = colours['y']
        self._braked = True
        return acceleration

    def _pass_left(self):
        self._target_lane = self._position[1] - self.LANE_W
        self._target_lane_ = self._target_lane_
        self._passing = True
        self._colour = colours['m']
        self._braked = False

    def _pass_right(self):
        self._target_lane = self._position[1] + self.LANE_W
        self._target_lane_ = self._target_lane_
        self._passing = True
        self._colour = colours['m']
        self._braked = False

    def __gt__(self, other):
        """
        Check if self is in front of other: self.back > other.front
        """
        return self.back > other.front

    def __lt__(self, other):
        """
        Check if self is behind of other: self.front < other.back
        """
        return self.front < other.back

    def __sub__(self, other):
        """
        Return the distance between self.back and other.front
        """
        return self.back - other.front

    def policy(self, observation):
        """
        Bring together _pass, brake
        :return: acceleration, d_theta
        """
        d_velocity_dt = 0

        car_ahead = observation[1][1]
        if car_ahead:
            distance = car_ahead - self
            if self.safe_distance > distance > 0:
                if random.random() < 0.5:
                    if self._safe_left(observation):
                        self._pass_left()
                    elif self._safe_right(observation):
                        self._pass_right()
                    else:
                        d_velocity_dt = self._brake(min((self.safe_distance / distance) ** 0.2 - 1, 1))
                else:
                    if self._safe_right(observation):
                        self._pass_right()
                    elif self._safe_left(observation):
                        self._pass_left()
                    else:
                        d_velocity_dt = self._brake(min((self.safe_distance / distance) ** 0.2 - 1, 1))

            elif distance <= 0:
                self._colour = colours['r']
                self.crashed = True

        if random.random() < 0.05:
            if self._safe_right(observation):
                self._pass_right()
                self._target_speed *= 0.95

        if d_velocity_dt == 0:
            d_velocity_dt = 1 * (self._target_speed - self._speed)

        if random.random() < 0.1:
            self._target_lane_ = self._target_lane + np.random.normal(0, self.LANE_W * 0.1)

        error = -(self._target_lane_ - self._position[1])
        d_error = error - self._error
        d_clip = 2
        if abs(d_error) > d_clip:
            d_error *= d_clip / abs(d_error)
        self._error = error
        ortho_direction = np.array((self._direction[1], -self._direction[0]))
        d_direction_dt = ortho_direction * (self.pid_k1 * error + self.pid_k2 * d_error)

        action = np.array((*d_direction_dt, d_velocity_dt))  # dx/dt, car state temporal derivative
        return action

    def _safe_left(self, state):
        if self.back < self.safe_distance: return False  # Cannot see in the future
        if self._passing: return False
        if state[0] is None: return False  # On the leftmost lane
        if state[0][0] and self - state[0][0] < state[0][0].safe_distance: return False
        if state[0][1] and state[0][1] - self < self.safe_distance: return False
        return True

    def _safe_right(self, state):
        if self.back < self.safe_distance: return False  # Cannot see in the future
        if self._passing: return False
        if state[2] is None: return False  # On the rightmost lane
        if state[2][0] and self - state[2][0] < state[2][0].safe_distance: return False
        if state[2][1] and state[2][1] - self < self.safe_distance: return False
        return True

    def _get_observation_image(self, m, screen_surface, width_height, scale):
        d = self._direction
        x_y = np.array((abs(d) @ width_height, abs(d) @ width_height[::-1]))
        centre = self._position + (self._length // 2, 0)
        sub_surface = screen_surface.subsurface((*(centre + m - x_y / 2), *x_y))
        theta = np.arctan2(*d[::-1]) * 180 / np.pi  # in degrees
        rot_surface = pygame.transform.rotate(sub_surface, theta)
        width_height = np.floor(np.array(width_height))
        x = (rot_surface.get_width() - width_height[0]) // 2
        y = (rot_surface.get_height() - width_height[1]) // 2
        sub_rot_surface = rot_surface.subsurface(x, y, *width_height)
        sub_rot_array = pygame.surfarray.array3d(sub_rot_surface).transpose(1, 0, 2)  # B channel not used
        sub_rot_array = scipy.misc.imresize(sub_rot_array, scale)
        sub_rot_array[:, :, 0] *= 4
        assert(sub_rot_array.max() <= 255.0)
        return torch.from_numpy(sub_rot_array)

    def store(self, object_name, object_):
        if object_name == 'action':
            self._actions.append(torch.Tensor(object_))
        elif object_name == 'state':
            self._states.append(self._get_obs(*object_))
        elif object_name == 'state_image':
            self._states_image.append(self._get_observation_image(*object_))

    def dump_state_image(self, save_dir='scratch/'):
        save_dir = save_dir + str(self.id)
        os.system('mkdir -p ' + save_dir)
        # im = self._states_image[100:]
        im = self._states_image
        for t in range(len(im)):
            imsave(f'{save_dir}/im{t:05d}.png', im[t].numpy())


class StatefulEnv(core.Env):

    # Environment's car class
    EnvCar = Car

    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W

    def __init__(self, display=True, nb_lanes=4, fps=30, delta_t=None, traffic_rate=15, state_image=True, store=True):

        self.offset = int(1.5 * self.LANE_W)
        self.screen_size = (80 * self.LANE_W, nb_lanes * self.LANE_W + self.offset + self.LANE_W // 2)
        self.fps = fps  # updates per second
        self.delta_t = delta_t or 1 / fps  # simulation timing interval
        self.nb_lanes = nb_lanes  # total number of lanes
        self.frame = 0  # frame index
        self.lanes = self.build_lanes(nb_lanes)  # create lanes object, list of dicts
        self.vehicles = None  # vehicles list
        self.traffic_rate = traffic_rate  # new cars per second
        self.lane_occupancy = None  # keeps track of what vehicle are in each lane
        self.collision = None  # an accident happened
        self.episode = 0  # episode counter
        self.car_id = None  # car counter init
        self.state_image = state_image
        self.mean_fps = None
        self.store = store
        self.policy_car_id = None
        self.next_car_id = None
        self.photos = None

        self.display = display
        if self.display:  # if display is required
            pygame.init()  # init PyGame
            self.screen = pygame.display.set_mode(self.screen_size)  # set screen size
            self.clock = pygame.time.Clock()  # set up timing

    def build_lanes(self, nb_lanes):
        return tuple(
            {'min': self.offset + n * self.LANE_W,
             'mid': self.offset + self.LANE_W / 2 + n * self.LANE_W,
             'max': self.offset + (n + 1) * self.LANE_W}
            for n in range(nb_lanes)
        )

    def reset(self):
        # Initialise environment state
        self.frame = 0
        self.vehicles = list()
        self.lane_occupancy = [[] for _ in self.lanes]
        self.episode += 1
        # keep track of the car we are controlling
        self.policy_car_id = -1
        self.next_car_id = 0
        self.mean_fps = None
        pygame.display.set_caption(f'Traffic simulator, episode {self.episode}')
        state = list()
        objects = list()
        return state, objects

    def step(self, policy_action=None):

        self.collision = False
        # Free lane beginnings
        # free_lanes = set(range(self.nb_lanes))
        free_lanes = set(range(1, self.nb_lanes))

        # For every vehicle
        #   t <- t + dt
        #   leave or enter lane
        #   remove itself if out of screen
        #   update free lane beginnings
        for v in self.vehicles[:]:
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
            if v.back > self.screen_size[0]:
                # if this is the controlled car, pick new car
                if v.id == self.policy_car_id:
                    self.policy_car_id = self.vehicles[-1].id
                for l in lanes_occupied: self.lane_occupancy[l].remove(v)
                self.vehicles.remove(v)

            # Update available lane beginnings
            if v.back < v.safe_distance:  # at most safe_distance ahead
                free_lanes -= lanes_occupied

        # Randomly add vehicles, up to 1 / dt per second
        if random.random() < self.traffic_rate * np.sin(2 * np.pi * self.frame * self.delta_t) * self.delta_t \
                or len(self.vehicles) == 0:
            if free_lanes:
                car = self.EnvCar(self.lanes, free_lanes, self.delta_t, self.next_car_id)
                self.next_car_id += 1
                self.vehicles.append(car)
                for l in car.get_lane_set(self.lanes):
                    # Prepend the new car to each lane it can be found
                    self.lane_occupancy[l].insert(0, car)

        if self.policy_car_id == -1:
            self.policy_car_id = 0

        if self.state_image:
            # How much to look far ahead
            look_ahead = MAX_SPEED * 1000 / 3600 * self.SCALE
            look_sideways = 2 * self.LANE_W
            self.render(mode='machine', width_height=(2 * look_ahead, 2 * look_sideways), scale=0.25)

        # Generate state representation for each vehicle
        for v in self.vehicles:
            lane_set = v.get_lane_set(self.lanes)
            # If v is in one lane only
            # Provide a list of (up to) 6 neighbouring vehicles
            current_lane_idx = lane_set.pop()
            # Given that I'm not in the left/right-most lane
            left_vehicles = self._get_neighbours(current_lane_idx, - 1, v) \
                if current_lane_idx > 0 and len(lane_set) == 0 else None
            mid_vehicles = self._get_neighbours(current_lane_idx, 0, v)
            right_vehicles = self._get_neighbours(current_lane_idx, + 1, v) \
                if current_lane_idx < len(self.lanes) - 1 else None

            state = left_vehicles, mid_vehicles, right_vehicles

            # Compute the action
            if v.id == self.policy_car_id and policy_action is not None:
                action = policy_action
            else:
                action = v.policy(state)

            # Check for accident
            if v.crashed: self.collision = v

            if self.store:
                v.store('state', state)
                v.store('action', action)

            # update the cars
            v.step(action)

        done = False

        if self.frame >= 10000:
            done = True

        if done:
            print(f'Episode ended, reward: {reward}, t={self.frame}')

        self.frame += 1

        obs = []
        # TODO: cost function
        cost = 0
        return obs, cost, done, self.vehicles

    def _get_neighbours(self, current_lane_idx, d_lane, v):
        target_lane = self.lane_occupancy[current_lane_idx + d_lane]
        # Find me in the lane
        if d_lane == 0:
            my_idx = target_lane.index(v)
        else:
            my_idx = bisect.bisect(target_lane, v)
        behind = target_lane[my_idx - 1] if my_idx > 0 else None
        if d_lane == 0: my_idx += 1
        ahead = target_lane[my_idx] if my_idx < len(target_lane) else None
        return behind, ahead

    def render(self, mode='human', width_height=None, scale=1.):
        if mode == 'human' and self.display:

            # self._pause()

            # capture the closing window and mouse-button-up event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONUP:
                    self._pause()
                    pdb.set_trace()

            # measure time elapsed, enforce it to be >= 1/fps
            fps = int(1 / self.clock.tick(self.fps) * 1e3)
            self.mean_fps = 0.9 * self.mean_fps + 0.1 * fps if self.mean_fps is not None else fps

            # clear the screen
            self.screen.fill(colours['k'])

            # background pictures
            if self.photos:
                for i in range(len(self.photos)):
                    self.screen.blit(self.photos[i], self.photos_rect[i])

            # draw lanes
            self._draw_lanes(self.screen)

            for v in self.vehicles:
                c = (v.id == self.policy_car_id)
                v.draw(self.screen, c)

            draw_text(self.screen, f'# cars: {len(self.vehicles)}', (10, 2))
            draw_text(self.screen, f'frame #: {self.frame}', (120, 2))
            draw_text(self.screen, f'fps: {self.mean_fps:.0f}', (270, 2))

            pygame.display.flip()

            # if self.collision: self._pause()

        if mode == 'machine':
            m = max_extension = np.linalg.norm(width_height)
            screen_surface = pygame.Surface(np.array(self.screen_size) + 2 * max_extension)

            # draw lanes
            self._draw_lanes(screen_surface, mode=mode, offset=max_extension)

            # draw vehicles
            for v in self.vehicles:
                v.draw(screen_surface, mode=mode, offset=max_extension)

            # extract states
            for i, v in enumerate(self.vehicles):
                if self.store:
                    v.store('state_image', (max_extension, screen_surface, width_height, scale))

    def _draw_lanes(self, surface, mode='human', offset=0):
        if mode == 'human':
            lanes = self.lanes
            for lane in lanes:
                sw = self.screen_size[0]  # screen width
                draw_dashed_line(surface, colours['w'], (0, lane['min']), (sw, lane['min']), 3)
                draw_dashed_line(surface, colours['r'], (0, lane['mid']), (sw, lane['mid']))
            pygame.draw.line(surface, colours['w'], (0, lanes[0]['min']), (sw, lanes[0]['min']), 3)
            pygame.draw.line(surface, colours['w'], (0, lanes[-1]['max']), (sw, lanes[-1]['max']), 3)
        if mode == 'machine':
            for lane in self.lanes:
                sw = self.screen_size[0] + 2 * offset  # screen width
                m = offset
                pygame.draw.line(surface, colours['r'], (0, lane['min'] + m), (sw, lane['min'] + m), 1)
                pygame.draw.line(surface, colours['r'], (0, lane['max'] + m), (sw, lane['max'] + m), 1)

    def _pause(self):
        pause = True
        while pause:
            self.clock.tick(15)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    sys.exit()
                elif e.type == pygame.MOUSEBUTTONUP:
                    pause = False
