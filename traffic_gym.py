import bisect

import pygame, pdb, torch
import math
import random
import numpy as np
import scipy.misc
import sys, pickle
from custom_graphics import draw_dashed_line, draw_text, draw_rect
from gym import core
import os
from imageio import imwrite
# from skimage.transform import rescale


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

    def __init__(self, lanes, free_lanes, dt, car_id, look_ahead, screen_w):
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
        self._noisy_target_lane = self._target_lane
        self.crashed = False
        self._error = 0
        self._states = list()
        self._states_image = list()
        self._actions = list()
        self._safe_factor = random.gauss(1, .2)  # 0.9 Germany, 2 safe
        self.pid_k1 = np.random.normal(1e-4, 1e-5)
        self.pid_k2 = np.random.normal(3e-3, 1e-4)
        self.look_ahead = look_ahead
        self.screen_w = screen_w

    def get_state(self):
        state = torch.zeros(4)
        state[0] = self._position[0]  # x
        state[1] = self._position[1]  # y
        state[2] = self._direction[0] * self._speed  # dx
        state[3] = self._direction[1] * self._speed  # dy
        return state

    def compute_cost(self, other):
        """
        Computes the cost associated with distance to the preceding vehicle
        :param other: the guy in front of me
        :return: cost
        """
        d = self._direction
        d_o = np.array((self._direction[1], -self._direction[0]))  # ortho direction, pointing left
        # max(0, .) required because my.front can > other.back
        cost_ahead = max(0, 1 - max(0, (other - self) @ d) / self.safe_distance)
        # abs() required because there are cars on the right too
        cost_sideways = max(0, 1 - abs((other - self) @ d_o) / self.LANE_W)

        return cost_ahead * cost_sideways

    def _get_obs(self, left_vehicles, mid_vehicles, right_vehicles):
        n_cars = 1 + 6  # this car + 6 neighbors
        obs = torch.zeros(n_cars, 2, 2)
        mask = torch.zeros(n_cars)
        obs = obs.view(n_cars, 4)
        cost = 0

        vstate = self.get_state()
        obs[0].copy_(vstate)

        if left_vehicles:
            if left_vehicles[0] is not None:
                s = left_vehicles[0].get_state()
                obs[1].copy_(s)
                mask[1] = 1
                cost = max(cost, left_vehicles[0].compute_cost(self))
            else:
                # for bag-of-cars this will be ignored by the mask,
                # but fill in with a similar value to not mess up batch norm
                obs[1].copy_(vstate)

            if left_vehicles[1] is not None:
                s = left_vehicles[1].get_state()
                obs[2].copy_(s)
                mask[2] = 1
                cost = max(cost, self.compute_cost(left_vehicles[1]))
            else:
                obs[2].copy_(vstate)
        else:
            obs[1].copy_(vstate)
            obs[2].copy_(vstate)

        if mid_vehicles[0] is not None:
            s = mid_vehicles[0].get_state()
            obs[3].copy_(s)
            mask[3] = 1
            cost = max(cost, mid_vehicles[0].compute_cost(self))
        else:
            obs[3].copy_(vstate)

        if mid_vehicles[1] is not None:
            s = mid_vehicles[1].get_state()
            obs[4].copy_(s)
            mask[4] = 1
            cost = max(cost, self.compute_cost(mid_vehicles[1]))
        else:
            obs[4].copy_(vstate)

        if right_vehicles:
            if right_vehicles[0] is not None:
                s = right_vehicles[0].get_state()
                obs[5].copy_(s)
                mask[5] = 1
                cost = max(cost, right_vehicles[0].compute_cost(self))
            else:
                obs[5].copy_(vstate)

            if right_vehicles[1] is not None:
                s = right_vehicles[1].get_state()
                obs[6].copy_(s)
                mask[6] = 1
                cost = max(cost, self.compute_cost(right_vehicles[1]))
            else:
                obs[6].copy_(vstate)
        else:
            obs[5].copy_(vstate)
            obs[6].copy_(vstate)

        # self._colour = (255 * cost, 0, 255 * (1 - cost))

        return obs, mask, cost

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

        # Hack... clip direction between -10 and +10
        d = self._direction
        # alpha = np.clip(np.arctan2(*d[::-1]), -10 * np.pi / 180, 10 * np.pi / 180)
        # d = np.array((np.cos(alpha), np.sin(alpha)))

        if mode == 'human':
            if c:
                pygame.draw.rect(surface, (0, 255, 0),
                                 (int(x - 15), int(y - 15), self._length + 20, self._width + 20), 2)
            draw_rect(surface, self._colour, rectangle, d, 3)
            draw_text(surface, str(self.id), (x, y - self._width // 2), 20, colours['b'])
            if self._braked: self._colour = colours['g']
        if mode == 'machine':
            draw_rect(surface, colours['g'], rectangle, d)

    def step(self, action):  # takes also the parameter action = state temporal derivative
        """
        Update current position, given current velocity and acceleration
        """
        # Actions: acceleration (a), steering (b)
        a, b = action

        # State integration
        self._position += self._speed * self._direction * self._dt

        ortho_direction = np.array((self._direction[1], -self._direction[0]))
        direction_vector = self._direction + ortho_direction * b * self._speed * self._dt
        self._direction = direction_vector / (np.linalg.norm(direction_vector) + 1e-3)

        self._speed += a * self._dt

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
        return self._speed * self._safe_factor + 1 * self.SCALE  # plus one metre

    @property
    def front(self):
        return self._position + self._length * self._direction

    @property
    def back(self):
        return self._position

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
        self._noisy_target_lane = self._noisy_target_lane
        self._passing = True
        self._colour = colours['m']
        self._braked = False

    def _pass_right(self):
        self._target_lane = self._position[1] + self.LANE_W
        self._noisy_target_lane = self._noisy_target_lane
        self._passing = True
        self._colour = colours['m']
        self._braked = False

    def __gt__(self, other):
        """
        Check if self is in front of other: self.back[0] > other.front[0]
        """
        return self.back[0] > other.front[0]

    def __lt__(self, other):
        """
        Check if self is behind of other: self.front[0] < other.back[0]
        """
        return self.front[0] < other.back[0]

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
        a = 0

        car_ahead = observation[1][1]
        if car_ahead:
            distance = (car_ahead - self)[0]
            if self.safe_distance > distance > 0:
                if random.random() < 0.5:
                    if self._safe_left(observation):
                        self._pass_left()
                    elif self._safe_right(observation):
                        self._pass_right()
                    else:
                        a = self._brake(min((self.safe_distance / distance) ** 0.2 - 1, 1))
                else:
                    if self._safe_right(observation):
                        self._pass_right()
                    elif self._safe_left(observation):
                        self._pass_left()
                    else:
                        a = self._brake(min((self.safe_distance / distance) ** 0.2 - 1, 1))

            elif distance <= 0:
                self._colour = colours['r']
                self.crashed = True

        if random.random() < 0.05:
            if self._safe_right(observation):
                self._pass_right()
                self._target_speed *= 0.95

        if a == 0:
            a = 1 * (self._target_speed - self._speed)

        if random.random() < 0.1:
            self._noisy_target_lane = self._target_lane + np.random.normal(0, LANE_W * 0.1)

        if random.random() < 0.05 and not self._passing:
            self._target_speed *= (1 + np.random.normal(0, 0.05))

        error = -(self._noisy_target_lane - self._position[1])
        d_error = error - self._error
        d_clip = 2
        if abs(d_error) > d_clip:
            d_error *= d_clip / abs(d_error)
        self._error = error
        b = self.pid_k1 * error + self.pid_k2 * d_error
        action = np.array((a, b))  # dx/dt, car state temporal derivative
        return action

    def _safe_left(self, state):
        if self.back[0] < self.safe_distance: return False  # Cannot see in the future
        if self._passing: return False
        if state[0] is None: return False  # On the leftmost lane
        if state[0][0] and (self - state[0][0])[0] < state[0][0].safe_distance: return False
        if state[0][1] and (state[0][1] - self)[0] < self.safe_distance: return False
        return True

    def _safe_right(self, state):
        if self.back[0] < self.safe_distance: return False  # Cannot see in the future
        if self._passing: return False
        if state[2] is None: return False  # On the rightmost lane
        if state[2][0] and (self - state[2][0])[0] < state[2][0].safe_distance: return False
        if state[2][1] and (state[2][1] - self)[0] < self.safe_distance: return False
        return True

    def _get_observation_image(self, m, screen_surface, width_height, scale):
        d = self._direction

        # Hack... clip direction between -10 and +10
        # alpha = np.clip(np.arctan2(*d[::-1]), -10 * np.pi / 180, 10 * np.pi / 180)
        # d = np.array((np.cos(alpha), np.sin(alpha)))

        x_y = np.ceil(np.array((abs(d) @ width_height, abs(d) @ width_height[::-1])))
        centre = self._position + (self._length // 2, 0)
        sub_surface = screen_surface.subsurface((*(centre + m - x_y / 2), *x_y))
        theta = np.arctan2(*d[::-1]) * 180 / np.pi  # in degrees
        rot_surface = pygame.transform.rotate(sub_surface, theta)
        width_height = np.floor(np.array(width_height))
        x = (rot_surface.get_width() - width_height[0]) // 2
        y = (rot_surface.get_height() - width_height[1]) // 2
        sub_rot_surface = rot_surface.subsurface(x, y, *width_height)
        sub_rot_array = pygame.surfarray.array3d(sub_rot_surface).transpose(1, 0, 2)  # B channel not used
        # sub_rot_array_scaled = rescale(sub_rot_array, scale, mode='constant')  # output not consistent with below
        sub_rot_array_scaled = scipy.misc.imresize(sub_rot_array, scale)  # is deprecated, need to be replaced
        sub_rot_array_scaled_up = np.rot90(sub_rot_array_scaled)  # facing upward, not flipped
        sub_rot_array_scaled_up[:, :, 0] *= 4
        assert sub_rot_array_scaled_up.max() <= 255

        # Compute cost relative to position within the lane
        x = np.ceil((rot_surface.get_width() - self._length) / 2)
        y = np.ceil((rot_surface.get_height() - self.LANE_W) / 2)
        neighbourhood = rot_surface.subsurface(x, y, self._length, self.LANE_W)
        neighbourhood_array = pygame.surfarray.array3d(neighbourhood).transpose(1, 0, 2)  # flip x and y
        lanes = neighbourhood_array[:, :, 0]
        mask = np.broadcast_to((1 - abs(np.linspace(-1, 1, self.LANE_W))).reshape(-1, 1), lanes.shape)
        lane_cost = (lanes * mask).max() / 255

        # self._colour = (255 * lane_cost, 0, 255 * (1 - lane_cost))

        return torch.from_numpy(sub_rot_array_scaled_up.copy()), lane_cost

    def store(self, object_name, object_):
        if object_name == 'action':
            self._actions.append(torch.Tensor(object_))
        elif object_name == 'state':
            self._states.append(self._get_obs(*object_))
        elif object_name == 'state_image':
            self._states_image.append(self._get_observation_image(*object_))

    def dump_state_image(self, save_dir='scratch/', mode='img'):
        os.system('mkdir -p ' + save_dir)
        # im = self._states_image[100:]
        transpose = list(zip(*self._states_image))
        im = transpose[0]
        lane_cost = transpose[1]
        os.system('mkdir -p ' + save_dir)
        if mode == 'tensor':
            # save in torch format
            im_pth = torch.stack(im).permute(0, 3, 1, 2)
            pickle.dump({
                'images': im_pth,
                'actions': torch.stack(self._actions),
                'lane_cost': torch.Tensor(lane_cost)
            }, open(save_dir + f'/car{self.id}.pkl', 'wb'))
        elif mode == 'img':
            save_dir = save_dir + '/' + str(self.id)
            os.system('mkdir -p ' + save_dir)
            for t in range(len(im)):
                imwrite(f'{save_dir}/im{t:05d}.png', im[t].numpy())

    @property
    def valid(self):
        return self.back[0] > self.look_ahead and self.front[0] < self.screen_w - 1.75 * self.look_ahead


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
        self.look_ahead = MAX_SPEED * 1000 / 3600 * self.SCALE
        self.look_sideways = 2 * self.LANE_W

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
            if v.back[0] > self.screen_size[0]:
                # if this is the controlled car, pick new car
                if v.id == self.policy_car_id:
                    self.policy_car_id = self.vehicles[-1].id
                for l in lanes_occupied: self.lane_occupancy[l].remove(v)
                self.vehicles.remove(v)

            # Update available lane beginnings
            if v.back[0] < v.safe_distance:  # at most safe_distance ahead
                free_lanes -= lanes_occupied

        # Randomly add vehicles, up to 1 / dt per second
        if random.random() < self.traffic_rate * np.sin(2 * np.pi * self.frame * self.delta_t) * self.delta_t or len(
                self.vehicles) == 0:
            if free_lanes:
                car = self.EnvCar(self.lanes, free_lanes, self.delta_t, self.next_car_id,
                                  self.look_ahead, self.screen_size[0])
                self.next_car_id += 1
                self.vehicles.append(car)
                for l in car.get_lane_set(self.lanes):
                    # Prepend the new car to each lane it can be found
                    self.lane_occupancy[l].insert(0, car)

        if self.policy_car_id == -1:
            self.policy_car_id = 0

        if self.state_image:
            self.render(mode='machine', width_height=(2 * self.look_ahead, 2 * self.look_sideways), scale=0.25)

        # Generate state representation for each vehicle
        for v in self.vehicles:
            lane_set = v.get_lane_set(self.lanes)
            # If v is in one lane only
            # Provide a list of (up to) 6 neighbouring vehicles
            if len(lane_set) == 0:
                lanes_occupied = v.get_lane_set(self.lanes)
                for l in lanes_occupied: self.lane_occupancy[l].remove(v)
                self.vehicles.remove(v)
                continue

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

            for place in state:
                if place is not None:
                    for car in place:
                        if car is not None:
                            s1 = car.get_state()
                            s2 = v.get_state()
                            if abs(s1[0] - s2[0]) < v._length and abs(s1[1] - s2[1]) < v._width:
                                v.crashed = True

            # Check for accident
            if v.crashed: self.collision = v

            if self.store and v.valid or v.id == self.policy_car_id:
                v.store('state', state)
                v.store('action', action)

            # update the cars
            v.step(action)

        '''
        for v in self.vehicles:
            cost = v._states[-1][2]
            if cost > 0.2:
                img = v._states_image[-1]
                hsh = random.random()
                imwrite(f'cost_images/high/im{hsh:.5f}_cost{cost}.png', img.numpy())
            elif cost < 0.01 and random.random() < 0.01:
                img = v._states_image[-1]
                hsh = random.random()
                imwrite(f'cost_images/low/im{hsh:.5f}_cost{cost}.png', img.numpy())
        '''

        self.frame += 1

        obs = []
        cost = 0
        return obs, cost, self.vehicles

    def _get_neighbours(self, current_lane_idx, d_lane, v):
        # Shallow copy the target lane
        target_lane = self.lane_occupancy[current_lane_idx + d_lane][:]
        # If I find myself in the target list, remove me
        if v in target_lane: target_lane.remove(v)
        # Find me in the lane
        my_idx = bisect.bisect(target_lane, v)
        behind = target_lane[my_idx - 1] if my_idx > 0 else None
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
                    # pdb.set_trace()

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
            vehicle_surface = pygame.Surface(np.array(self.screen_size) + 2 * max_extension)

            # draw lanes
            self._draw_lanes(screen_surface, mode=mode, offset=max_extension)

            # draw vehicles
            for v in self.vehicles:
                v.draw(vehicle_surface, mode=mode, offset=max_extension)

            screen_surface.blit(vehicle_surface, (0, 0), special_flags=pygame.BLEND_ADD)

            # extract states
            for i, v in enumerate(self.vehicles):
                if self.store and v.valid or v.id == self.policy_car_id:
                    v.store('state_image', (max_extension, screen_surface, width_height, scale))

    def _draw_lanes(self, surface, mode='human', offset=0):
        draw_line = pygame.draw.line
        if mode == 'human':
            lanes = self.lanes
            for lane in lanes:
                sw = self.screen_size[0]  # screen width
                draw_dashed_line(surface, colours['w'], (0, lane['min']), (sw, lane['min']), 3)
                draw_dashed_line(surface, colours['r'], (0, lane['mid']), (sw, lane['mid']))
            draw_line(surface, colours['w'], (0, lanes[0]['min']), (sw, lanes[0]['min']), 3)
            bottom = lanes[-1]['max']
            draw_line(surface, colours['w'], (0, bottom), (sw, bottom), 3)

            look_ahead = MAX_SPEED * 1000 / 3600 * self.SCALE
            o = self.offset
            draw_line(surface, (255, 255, 0), (look_ahead, o), (look_ahead, 9.4 * LANE_W))
            draw_line(surface, (255, 255, 0), (sw - 1.75 * look_ahead, o), (sw - 1.75 * look_ahead, bottom))
            draw_line(surface, (255, 255, 0), (sw - 0.75 * look_ahead, o), (sw - 0.75 * look_ahead, bottom), 5)
        if mode == 'machine':
            for lane in self.lanes:
                sw = self.screen_size[0] + 2 * offset  # screen width
                m = offset
                draw_line(surface, colours['r'], (0, lane['min'] + m), (sw, lane['min'] + m), 1)
                draw_line(surface, colours['r'], (0, lane['max'] + m), (sw, lane['max'] + m), 1)

    def _pause(self):
        pause = True
        while pause:
            self.clock.tick(15)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    sys.exit()
                elif e.type == pygame.MOUSEBUTTONUP:
                    pause = False
