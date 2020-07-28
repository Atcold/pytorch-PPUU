import bisect

import pygame, pdb, torch
import math, numpy
import random
import numpy as np
import scipy.misc
import sys, pickle
# from skimage import measure, transform
# from matplotlib.image import imsave
import PIL
from PIL import Image
from custom_graphics import draw_dashed_line, draw_text, draw_rect
from gym import core, spaces
import os
from imageio import imwrite

# from skimage.transform import rescale


# Conversion LANE_W from real world to pixels
# A US highway lane width is 3.7 metres, here 50 pixels
LANE_W = 24  # pixels / 3.7 m, lane width
SCALE = LANE_W / 3.7  # pixels per metre

STATE_C = 3
STATE_H, STATE_W = 117, 24
STATE_D = 4

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

# Car coordinate system, origin under the centre of the rear axis
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

    def __init__(self, lanes, free_lanes, dt, car_id, look_ahead, screen_w, font, policy_type, policy_network=None):
        """
        Initialise a sedan on a random lane
        :param lanes: tuple of lanes, with ``min`` and ``max`` y coordinates
        :param dt: temporal updating interval
        """
        self._length = round(4.8 * self.SCALE)
        self._width = round(1.8 * self.SCALE)
        self.id = car_id
        lane = random.choice(tuple(free_lanes))
        if lane == 6 and type(self).__name__ == 'PatchedCar':
            self._position = np.array((0, lanes[-1]['max'] + 42), np.float)
            self._direction = np.array((1, -0.035), np.float) / np.sqrt(1 + 0.035 ** 2)
        else:
            self._position = np.array((
                -self._length,
                lanes[lane]['mid']
            ), np.float)
            self._direction = np.array((1, 0), np.float)
        self._target_speed = max(
            0,
            (MAX_SPEED - random.randrange(0, 15) - 10 * lane)
        ) * 1000 / 3600 * self.SCALE  # m / s
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
        self._ego_car_image = None
        self._actions = list()
        self._safe_factor = random.gauss(1.5, 0)  # 0.9 Germany, 2 safe
        self.pid_k1 = np.random.normal(1e-4, 1e-5)
        self.pid_k2 = np.random.normal(1e-3, 1e-4)
        self.look_ahead = look_ahead
        self.screen_w = screen_w
        self._text = self.get_text(self.id, font)
        self._policy_type = policy_type
        self.policy_network = policy_network
        self.is_controlled = False
        self.collisions_per_frame = 0

    @staticmethod
    def get_text(n, font):
        text = font.render(str(n), True, colours['b'])
        text_rect = text.get_rect()
        return text, text_rect

    def get_state(self):
        state = torch.zeros(4)
        state[0] = self._position[0]  # x
        state[1] = self._position[1]  # y
        state[2] = self._direction[0] * self._speed  # dx/dt
        state[3] = self._direction[1] * self._speed  # dy/dt
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

        v_state = self.get_state()
        obs[0].copy_(v_state)

        if left_vehicles:
            if left_vehicles[0] is not None:
                s = left_vehicles[0].get_state()
                obs[1].copy_(s)
                mask[1] = 1
                cost = max(cost, left_vehicles[0].compute_cost(self))
            else:
                # for bag-of-cars this will be ignored by the mask,
                # but fill in with a similar value to not mess up batch norm
                obs[1].copy_(v_state)

            if left_vehicles[1] is not None:
                s = left_vehicles[1].get_state()
                obs[2].copy_(s)
                mask[2] = 1
                cost = max(cost, self.compute_cost(left_vehicles[1]))
            else:
                obs[2].copy_(v_state)
        else:
            obs[1].copy_(v_state)
            obs[2].copy_(v_state)

        if mid_vehicles[0] is not None:
            s = mid_vehicles[0].get_state()
            obs[3].copy_(s)
            mask[3] = 1
            cost = max(cost, mid_vehicles[0].compute_cost(self))
        else:
            obs[3].copy_(v_state)

        if mid_vehicles[1] is not None:
            s = mid_vehicles[1].get_state()
            obs[4].copy_(s)
            mask[4] = 1
            cost = max(cost, self.compute_cost(mid_vehicles[1]))
        else:
            obs[4].copy_(v_state)

        if right_vehicles:
            if right_vehicles[0] is not None:
                s = right_vehicles[0].get_state()
                obs[5].copy_(s)
                mask[5] = 1
                cost = max(cost, right_vehicles[0].compute_cost(self))
            else:
                obs[5].copy_(v_state)

            if right_vehicles[1] is not None:
                s = right_vehicles[1].get_state()
                obs[6].copy_(s)
                mask[6] = 1
                cost = max(cost, self.compute_cost(right_vehicles[1]))
            else:
                obs[6].copy_(v_state)
        else:
            obs[5].copy_(v_state)
            obs[6].copy_(v_state)

        # self._colour = (255 * cost, 0, 255 * (1 - cost))
        # if cost and cost > 0.95:
        #     print(f'Car {self.id} prox cost: {cost:.2f}')

        return obs, mask, cost

    def draw(self, surface, mode='human', offset=0):
        """
        Draw current car on screen with a specific colour
        :param surface: PyGame ``Surface`` where to draw
        :param mode: human or machine
        :param offset: for representation cropping
        """
        x, y = self._position + offset
        rectangle = (int(x), int(y), self._length, self._width)

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

            _r = draw_rect(surface, self._colour, rectangle, d)

            # Drawing vehicle number
            if x < self.front[0]:
                self._text[1].left = x
            else:
                self._text[1].right = x
            self._text[1].top = y - self._width // 2
            surface.blit(self._text[0], self._text[1])

            if self._braked: self._colour = colours['g']
            return _r
        if mode == 'machine':
            return draw_rect(surface, colours['g'], rectangle, d)
        if mode == 'ego-car':
            return draw_rect(surface, colours['b'], rectangle, d)
        if mode == 'ghost':
            return draw_rect(surface, colours['y'], rectangle, d)

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
        return abs(self._speed) * self._safe_factor + 1 * self.SCALE  # plus one metre

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
        Check if self is in front of other: self.front[0] > other.front[0]
        """
        return self.front[0] > other.front[0]

    def __lt__(self, other):
        """
        Check if self is behind of other: self.front[0] < other.front[0]
        """
        return self.front[0] < other.front[0]

    def __sub__(self, other):
        """
        Return the distance between self.back and other.front
        """
        return self.back - other.front

    def policy(self, observation, policy_type):
        if policy_type == 'hardcoded':
            return self.policy_hardcoded(observation)
        elif policy_type == 'imitation':
            return self.policy_imitation(observation)

    def policy_hardcoded(self, observation):
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

        # if random.random() < 0.1:
        #     self._noisy_target_lane = self._target_lane + np.random.normal(0, LANE_W * 0.1)
        # error = -(self._noisy_target_lane - self._position[1])

        # if random.random() < 0.05 and not self._passing:
        #     self._target_speed *= (1 + np.random.normal(0, 0.05))

        error = -(self._target_lane - self._position[1])
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

    def _get_observation_image(self, m, screen_surface, width_height, scale, global_frame):
        d = self._direction

        x_y = np.ceil(np.array((abs(d) @ width_height, abs(d) @ width_height[::-1])))
        centre = self._position + (d * self._length) // 2
        try:
            sub_surface = screen_surface.subsurface((*(centre + m - x_y / 2), *x_y))
        except ValueError as ex:  # if the agent fucks up
            print(f'{self} fucked up')  # notify about the event
            self.off_screen = True  # we're off_screen
            return self._states_image[-1]  # return last state
        theta = np.arctan2(*d[::-1]) * 180 / np.pi  # in degrees
        rot_surface = pygame.transform.rotate(sub_surface, theta)
        width_height = np.floor(np.array(width_height))
        surf_w = rot_surface.get_width()
        surf_h = rot_surface.get_height()
        x = (surf_w - width_height[0]) // 2
        y = (surf_h - width_height[1]) // 2
        sub_rot_surface = rot_surface.subsurface(x, y, *width_height)
        sub_rot_array = pygame.surfarray.array3d(sub_rot_surface).transpose(1, 0, 2)  # flip x and y
        # sub_rot_array_scaled = rescale(sub_rot_array, scale, mode='constant')  # output not consistent with below
        new_h = int(scale*sub_rot_array.shape[0])
        new_w = int(scale*sub_rot_array.shape[1])
        sub_rot_array_scaled = np.array(PIL.Image.fromarray(sub_rot_array).resize((new_w, new_h), resample=2)) #bilinear
        sub_rot_array_scaled_up = np.rot90(sub_rot_array_scaled)  # facing upward, not flipped
        sub_rot_array_scaled_up[:, :, 0] *= 4
        assert sub_rot_array_scaled_up.max() <= 255

        # Compute cost relative to position within the lane
        x = np.ceil((surf_w - self._length) / 2)
        y = np.ceil((surf_h - self.LANE_W) / 2)
        neighbourhood = rot_surface.subsurface(x, y, self._length, self.LANE_W)
        neighbourhood_array = pygame.surfarray.array3d(neighbourhood).transpose(1, 0, 2)  # flip x and y
        lanes = neighbourhood_array[:, :, 0]
        lane_mask = np.broadcast_to((1 - abs(np.linspace(-1, 1, self.LANE_W))).reshape(-1, 1), lanes.shape)
        lane_cost = (lanes * lane_mask).max() / 255

        # Compute x/y minimum distance to other vehicles (pixel version)
        # Account for 1 metre overlap (low data accuracy)
        alpha = 1 * self.SCALE  # 1 m overlap collision
        # Create separable proximity mask
        crop_h, crop_w, _ = sub_rot_array.shape
        max_x = np.ceil((crop_w - max(self._length - alpha, 0)) / 2)
        max_y = np.ceil((crop_h - max(self._width - alpha, 0)) / 2)
        min_x = max(np.ceil(max_x - self.safe_distance), 0)
        min_y = np.ceil(crop_h / 2 - self._width)  # assumes other._width / 2 = self._width / 2
        x_filter = (1 - abs(np.linspace(-1, 1, crop_w))) * crop_w / 2  # 45 degree
        x_filter[x_filter > max_x] = max_x  # chop off top
        x_filter[x_filter < min_x] = min_x  # chop off bottom
        x_filter = (x_filter - min_x) / (max_x - min_x)  # normalise
        y_filter = (1 - abs(np.linspace(-1, 1, crop_h))) * crop_h / 2  # 45 degree
        y_filter[y_filter > max_y] = max_y  # chop off top
        y_filter[y_filter < min_y] = min_y  # chop off bottom
        y_filter = (y_filter - min_y) / (max_y - min_y)  # normalise
        proximity_mask = y_filter.reshape(-1, 1) @ x_filter.reshape(1, -1)
        # Compute cost
        vehicles = sub_rot_array[:, :, 1]  # flip x and y, get green
        proximity_cost = (vehicles * proximity_mask).max() / 255

        # Inspecting collisions
        # if proximity_cost > 0.99:
        #     with open(f'scratch/collisions/{self}-{self._frame}.pkl', 'wb') as f:
        #         pickle.dump({
        #             'vehicles': vehicles,
        #             'proximity_mask': proximity_mask,
        #             'proximity_cost': proximity_cost,
        #             'sub_rot_array': sub_rot_array,
        #         }, f)

        # # Draw boxes, for visualisation purpose
        # # init as: env.reset(time_interval=1, frame=2510, control=False)
        # if self.id in (1033, 987, 992, 958, 961):
        #     w, h = width_height
        #     points = np.array(((w, -h), (-w, -h), (-w, h), (w, h))) / 2
        #     c, s = d
        #     rot = np.array(((c, -s), (s, c)))
        #     rot_points = (rot @ points.T).T + centre + m
        #     pygame.draw.polygon(screen_surface, colours['c'], rot_points, 1)
        #     imsave(f'car {self.id}.png', sub_rot_array_scaled_up)

        # self._colour = (255 * lane_cost, 0, 255 * (1 - lane_cost))

        # return state_image, lane_cost, proximity_cost, frame
        return torch.from_numpy(sub_rot_array_scaled_up.copy()), lane_cost, proximity_cost, global_frame

    def store(self, object_name, object_):
        if object_name == 'action':
            self._actions.append(torch.Tensor(object_))
        elif object_name == 'state':
            self._states.append(self._get_obs(*object_))
        elif object_name == 'state_image':
            self._states_image.append(self._get_observation_image(*object_))
        elif object_name == 'ego_car_image' and self._ego_car_image is None:
            self._ego_car_image = self._get_observation_image(*object_)[0]

    def get_last(self, n, done, norm_state=False, return_reward=False, gamma=0.99):
        if len(self._states_image) < n: return None  # no enough samples
        # n × (state_image, lane_cost, proximity_cost, frame) ->
        # -> (n × state_image, n × lane_cost, n × proximity_cost, n × frame)
        transpose = list(zip(*self._states_image))
        state_images = transpose[0]
        state_images = torch.stack(state_images).permute(0, 3, 1, 2)[-n:]
        ego_car_new_shape = list(state_images.shape)
        ego_car_new_shape[1] = 1
        ego_car_channel = self._ego_car_image[:, :, 2][None, None, :].expand(ego_car_new_shape)
        state_images = torch.cat((state_images, ego_car_channel), 1)

        zip_ = list(zip(*self._states))  # n × (obs, mask, cost) -> (n × obs, n × mask, n × cost)
        states = torch.stack(zip_[0])[:, 0][-n:]  # select the ego-state (of 1 + 6 states we keep track)
        if norm_state is not False:  # normalise the states, if requested
            states = states.sub(norm_state['s_mean']).div(norm_state['s_std'])  # N(0, 1) range
            state_images = state_images.float().div(255)  # [0, 1] range
        observation = dict(context=state_images, state=states)

        cost = dict(
            proximity_cost=self._states[-1][2],
            lane_cost=self._states_image[-1][1],
            pixel_proximity_cost=self._states_image[-1][2],
            collisions_per_frame=self.collisions_per_frame,
            arrived_to_dst=self.arrived_to_dst,
        )

        if return_reward:  # if we're playing with model free RL, have fun with reward shaping
            arrived = self.arrived_to_dst
            collision = self.collisions_per_frame > 0
            done = done or collision  # die if collide
            lambda_lane = 0.2
            max_rew = 1 + lambda_lane
            win = max_rew / (1 - gamma)
            reward = max_rew - cost['pixel_proximity_cost'] - lambda_lane * cost['lane_cost'] + win * arrived

            # So, observation must be just one damn numpy thingy
            observation = torch.cat((
                states.view(n, -1),
                state_images.view(n, -1),
            ), dim=1).numpy()

            return observation, reward, self.off_screen or done, dict(v=str(self), a=self.arrived_to_dst)

        return observation, cost, self.off_screen or done, self

    def dump_state_image(self, save_dir='scratch/', mode='img'):
        os.system('mkdir -p ' + save_dir)
        transpose = list(zip(*self._states_image))
        if len(transpose) == 0:
            print(f'failure, {save_dir}')
            # print(transpose)
            return
        im = transpose[0]
        if mode == 'tensor':
            lane_cost = torch.Tensor(transpose[1])
            pixel_proximity_cost = torch.Tensor(transpose[2])
            frames = np.array(transpose[3])
            zip_ = list(zip(*self._states))
            proximity_cost = torch.Tensor(zip_[2])
            states = torch.stack(zip_[0])
            mask = torch.stack(zip_[1])
            # save in torch format
            im_pth = torch.stack(im).permute(0, 3, 1, 2)
            with open(os.path.join(save_dir, f'car{self.id}.pkl'), 'wb') as f:
                pickle.dump({
                    'images': im_pth,
                    'actions': torch.stack(self._actions),
                    'lane_cost': lane_cost,
                    'pixel_proximity_cost': pixel_proximity_cost,
                    'states': states,
                    'proximity_cost': proximity_cost,
                    'mask': mask,
                    'frames': frames,
                    'ego_car': self._ego_car_image.permute(2, 0, 1),
                }, f)
        elif mode == 'img':
            save_dir = os.path.join(save_dir, str(self.id))
            os.system('mkdir -p ' + save_dir)
            for t in range(len(im)):
                imwrite(f'{save_dir}/im{t:05d}.png', im[t].numpy())

    @property
    def valid(self):
        return self.back[0] > self.look_ahead and self.front[0] < self.screen_w - 1.75 * self.look_ahead

    def __repr__(self) -> str:
        cls = self.__class__
        return f'{cls.__module__}.{cls.__name__}.{self.id}'

    @property
    def shape(self):
        return self._length, self._width


class Simulator(core.Env):
    # Environment's car class
    EnvCar = Car

    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W
    DUMP_NAME = 'data_ai_v0'

    # Action space definition
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # brake / accelerate, right / left

    def __init__(self, display=True, nb_lanes=4, fps=30, delta_t=None, traffic_rate=15, state_image=False, store=False,
                 policy_type='hardcoded', nb_states=0, data_dir='', normalise_action=False, normalise_state=False,
                 return_reward=False, gamma=0.99, show_frame_count=True, store_simulator_video=False):

        # Observation spaces definition
        self.observation_space = spaces.Box(low=-1, high=1, shape=(nb_states, STATE_D + STATE_C * STATE_H * STATE_W), dtype=np.float32)

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
        self.state_image = state_image or policy_type == 'imitation'
        self.mean_fps = None
        self.store = store or policy_type == 'imitation'
        self.next_car_id = None
        self.photos = None
        self.look_ahead = MAX_SPEED * 1000 / 3600 * self.SCALE
        self.look_sideways = 2 * self.LANE_W
        self.policy_type = policy_type
        self.actions_buffer = []
        self.policy_network = None
        self._lane_surfaces = dict()
        self.time_counter = None
        self.controlled_car = None
        self.nb_states = nb_states
        self.data_dir = data_dir
        self.user_is_done = None

        self.display = display
        if self.display:  # if display is required
            pygame.init()  # init PyGame
            self.screen = pygame.display.set_mode(self.screen_size)  # set screen size
            self.clock = pygame.time.Clock()  # set up timing
            self.font = {
                20: pygame.font.SysFont(None, 20),
                30: pygame.font.SysFont(None, 30),
            }

        self.random = random.Random()
        self.normalise_action = normalise_action
        self.normalise_state = normalise_state
        self.return_reward = return_reward
        self.gamma = gamma
        self.done = None
        self.show_frame_count = show_frame_count
        self.ghost = None
        self.store_sim_video = store_simulator_video

    def seed(self, seed=None):
        self.random.seed(seed)

    def build_lanes(self, nb_lanes):
        return tuple(
            {'min': self.offset + n * self.LANE_W,
             'mid': self.offset + self.LANE_W / 2 + n * self.LANE_W,
             'max': self.offset + (n + 1) * self.LANE_W}
            for n in range(nb_lanes)
        )

    def set_policy(self, policy_network):
        self.policy_network = policy_network

    def reset(self, control=True, **kwargs):
        # Initialise environment state
        self.frame = 0
        self.vehicles = list()
        self.lane_occupancy = [[] for _ in range(self.nb_lanes)]
        self.episode += 1
        # keep track of the car we are controlling
        self.next_car_id = 0
        self.mean_fps = None
        self.time_counter = 0
        pygame.display.set_caption(f'Traffic simulator, episode {self.episode}, start from frame {self.frame}')
        if control:
            self.controlled_car = {
                'locked': False,
            }
        self.user_is_done = False
        self.done = False

    def policy_imitation(self, observation):
        s_mean = torch.Tensor([891.5662, 116.9270, 39.2255, -0.2574])
        s_std = torch.Tensor([391.5376, 43.8825, 25.1841, 1.0992])

        # observation is a tuple (images, states)
        images = observation[0].contiguous()
        states = observation[1].contiguous()
        images.div_(255.0)
        bsize = images.size(0)

        states -= s_mean.view(1, 1, 4).expand(states.size())
        states /= (1e-8 + s_std.view(1, 1, 4).expand(states.size()))

        images = images.float()
        states = states.float()
        _, _, _, actions = self.policy_network(images, states, sample=True, unnormalize=True)
        actions = actions.view(bsize, -1, 2)
        return actions

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
                for l in lanes_occupied: self.lane_occupancy[l].remove(v)
                self.vehicles.remove(v)

            # Update available lane beginnings
            if v.back[0] < v.safe_distance:  # at most safe_distance ahead
                free_lanes -= lanes_occupied

        # Randomly add vehicles, up to 1 / dt per second
        if random.random() < self.traffic_rate * np.sin(2 * np.pi * self.frame * self.delta_t) * self.delta_t:
            if free_lanes:
                car = self.EnvCar(self.lanes, free_lanes, self.delta_t, self.next_car_id,
                                  self.look_ahead, self.screen_size[0], self.font[20], policy_type=self.policy_type,
                                  policy_network=self.policy_network)
                self.next_car_id += 1
                self.vehicles.append(car)
                for l in car.get_lane_set(self.lanes):
                    # Prepend the new car to each lane it can be found
                    self.lane_occupancy[l].insert(0, car)

        if self.state_image:
            self.render(mode='machine', width_height=(2 * self.look_ahead, 2 * self.look_sideways), scale=0.25)

        # Generate state representation for each vehicle

        # remove vehicles that need to be removed first
        for v in self.vehicles:
            lane_set = v.get_lane_set(self.lanes)
            if len(lane_set) == 0:
                lanes_occupied = v.get_lane_set(self.lanes)
                for l in lanes_occupied: self.lane_occupancy[l].remove(v)
                self.vehicles.remove(v)

        states_images, states_raw, update = [], [], []
        # print(len(self.vehicles))
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

            if self.policy_type == 'imitation':
                if len(v._states_image) > 10:  # and v.id == self.policy_car_id:
                    state_image, state_raw = v.get_last(10)
                    v.update = 1
                else:
                    state_image, state_raw = torch.zeros(10, 3, 117, 24), torch.zeros(10, 4)
                    v.update = 0

                states_images.append(state_image.float())
                states_raw.append(state_raw.float())
                v.store('state', state)

            if self.policy_type == 'hardcoded':
                # Compute the action
                if v.is_controlled and policy_action is not None:
                    action = policy_action
                else:
                    # if len(v._states_image) >= 10 and self.policy_type == 'imitation':
                    #     state_ = v.get_last_state_image(10)
                    #     action = v.policy(state_, 'imitation')
                    #     # print('here')
                    # else:
                    #     # if len(v._states_image) > 15:
                    #     #     pdb.set_trace()
                    action = v.policy(state, 'hardcoded')

                # Check for accident
                if v.crashed: self.collision = v

                if (self.store or v.is_controlled) and v.valid:
                    v.store('state', state)
                    v.store('action', action)

                # update the cars
                v.step(action)

        if self.policy_type == 'imitation' and len(self.vehicles) > 0:
            # update the cars
            predictions_nb = 20
            if self.time_counter == 0 or len(self.vehicles) != self.actions_buffer.size(0):
                print('new actions')
                states_images = torch.stack(states_images)
                states_raw = torch.stack(states_raw)
                self.actions_buffer = self.policy_imitation([states_images, states_raw])
                self.time_counter = 0
            car_counter = 0
            for v in self.vehicles:
                if v.update == 1:
                    if car_counter >= self.actions_buffer.size(0):
                        pdb.set_trace()
                    action = self.actions_buffer[car_counter][self.time_counter].numpy()
                else:
                    action = np.array([0, 0])
                # print(action)
                # action = np.array([0, 0])
                b = action[1]
                action[1] = min(abs(b), v._speed / MAX_SPEED / SCALE * .01) * np.sign(b)
                v.step(action)
                # if v.id == 2:
                # print(v.id, *action, v._speed / SCALE, v._target_speed / SCALE)
                # v.store('action', action)
                car_counter += 1
            self.time_counter += 1
            if self.time_counter >= predictions_nb:
                self.time_counter = 0

        self.frame += 1

        # return observation, reward, done, info
        return None, None, False, self.vehicles

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

            # if self.frame % 1000 == 0:
            #     pygame.image.save(self.screen, "Peachtree/ghosts.png")
            #     self.screen.fill(colours['k'])
            #     self._pause()

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
                v.draw(self.screen)

            draw_text(self.screen, f'# cars: {len(self.vehicles)}', (10, 2), font=self.font[30])
            draw_text(self.screen, f'frame #: {self.frame}', (120, 2), font=self.font[30])
            draw_text(self.screen, f'fps: {self.mean_fps:.0f}', (270, 2), font=self.font[30])

            pygame.display.flip()

            # # save surface as image, for visualisation only
            # pygame.image.save(self.screen, "screen_surface.png")
            # pygame.image.save(self.screen, f'screen-dumps/{self.dump_folder}/{self.frame:08d}.png')

            # capture the closing window and mouse-button-up event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONUP:
                    self._pause()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                    self.user_is_done = True

            # if self.collision:
            #     self._pause()
            #     self.collision = False

        if mode == 'machine':
            max_extension = int(np.linalg.norm(width_height) / 2)
            machine_screen_size = np.array(self.screen_size) + 2 * max_extension
            vehicle_surface = pygame.Surface(machine_screen_size)

            # draw lanes
            try:
                lane_surface = self._lane_surfaces[mode]

            except KeyError:
                lane_surface = pygame.Surface(machine_screen_size)
                self._draw_lanes(lane_surface, mode=mode, offset=max_extension)

            # # draw vehicles
            # for v in self.vehicles:
            #     v.draw(vehicle_surface, mode=mode, offset=max_extension)
            #
            # vehicle_surface.blit(lane_surface, (0, 0), special_flags=pygame.BLEND_MAX)

            # extract states
            ego_surface = pygame.Surface(machine_screen_size)
            for i, v in enumerate(self.vehicles):
                if (self.store or v.is_controlled) and v.valid:
                    # For every vehicle we want to extract the state, start with a black surface
                    vehicle_surface.fill((0, 0, 0))
                    # Draw all the other vehicles (in green)
                    for vv in set(self.vehicles) - {v}:
                        vv.draw(vehicle_surface, mode=mode, offset=max_extension)
                    # Superimpose the lanes
                    vehicle_surface.blit(lane_surface, (0, 0), special_flags=pygame.BLEND_MAX)
                    # Empty ego-surface
                    ego_surface.fill((0, 0, 0))
                    # Draw myself blue on the ego_surface
                    ego_rect = v.draw(ego_surface, mode='ego-car', offset=max_extension)
                    # Add me on top of others without shadowing
                    # vehicle_surface.blit(ego_surface, ego_rect, ego_rect, special_flags=pygame.BLEND_MAX)
                    v.store('state_image', (max_extension, vehicle_surface, width_height, scale, self.frame))
                    v.store('ego_car_image', (max_extension, ego_surface, width_height, scale, self.frame))
                    # Store whole history, if requested
                    if self.store_sim_video:
                        if self.ghost:
                            self.ghost.draw(vehicle_surface, mode='ghost', offset=max_extension)
                        v.frames.append(pygame.surfarray.array3d(vehicle_surface).transpose(1, 0, 2))  # flip x and y

            # # save surface as image, for visualisation only
            # pygame.image.save(vehicle_surface, "vehicle_surface.png")
            # self._pause()

    def _draw_lanes(self, surface, mode='human', offset=0):
        draw_line = pygame.draw.line
        if mode == 'human':
            lanes = self.lanes
            sw = self.screen_size[0]  # screen width
            for lane in lanes:
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
            sw = self.screen_size[0] + 2 * offset  # screen width
            for lane in self.lanes:
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
                elif e.type == pygame.MOUSEBUTTONUP or e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                    pause = False

    def _get_vehicle(self, id_):
        return self.vehicles[[v.id for v in self.vehicles].index(id_)]
