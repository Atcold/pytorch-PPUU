import pygame
import math
import random
import numpy as np
import sys
from custom_graphics import draw_dashed_line, draw_text
from gym import core

seed = 123
random.seed(seed)
np.random.seed(seed)

# Conversion LANE_W from real world to pixels
# A US highway lane width is 3.7 metres, here 50 pixels
LANE_W = 20  # pixels / 3.7 m, lane width
SCALE = LANE_W / 3.7

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

class Car:
    def __init__(self, lanes, dt):
        """
        Initialise a sedan on a random lane
        :param lanes: tuple of lanes, with ``min`` and ``max`` y coordinates
        :param dt: temporal updating interval
        """
        self.length = round(4.8 * SCALE)
        self.width = round(1.8 * SCALE)
        self.direction = np.array((1, 0), np.float)
        self.position = np.array((
            -self.length,
            random.choice(lanes)['mid'] - self.width // 2
        ), np.float)
        self.target_speed = random.randrange(80, 120) * 1000 / 3600 * SCALE  # m / s
        self.speed = self.target_speed
        self.dt = dt
        self.acceleration = 0
        self.colour = colours['c']
        self._braked = False

    def draw(self, screen):
        """
        Draw current car on screen with a specific colour
        :param screen: PyGame ``Surface`` where to draw
        """
        x, y = self.position
        rectangle = (int(x), int(y), self.length, self.width)
        pygame.draw.rect(screen, self.colour, rectangle)
        pygame.draw.rect(screen, tuple(c/2 for c in self.colour), rectangle, 4)
        if self._braked: self.colour = colours['g']

    def step(self):  # takes also the parameter action = state temporal derivative
        """
        Update current position, given current velocity and acceleration
        """
        self.position += max(self.speed + self.acceleration * self.dt, 0) * self.direction * self.dt
        self.acceleration = 0

    def get_lane_set(self, lanes):
        """
        Returns the set of lanes currently occupied
        :param lanes: tuple of lanes, with ``min`` and ``max`` y coordinates
        :return: busy lanes set
        """
        busy_lanes = set()
        y = self.position[1]
        w = self.width
        for lane_idx, lane in enumerate(lanes):
            if lane['min'] <= y <= lane['max'] or lane['min'] <= y + w <= lane['max']:
                busy_lanes.add(lane_idx)
        return busy_lanes

    def safe_distance(self):
        factor = 2  # 0.9 Germany, 2 safe
        return self.speed * factor

    def front(self):
        return int(self.position[0] + self.length)

    def back(self):
        return int(self.position[0])

    def brake(self, fraction):
        g, mu = 9.81, 0.9  # gravity and friction coefficient
        self.acceleration = -fraction * g * mu * SCALE
        self.colour = colours['y']
        self._braked = True


class StatefulEnv(core.Env):

    def __init__(self, display=True, nb_lanes=4, fps=30):
        self.display = display
        self.screen_size = (80 * LANE_W, (nb_lanes + 1) * LANE_W)
        self.fps = fps
        self.delta_t = 1 / fps
        self.nb_lanes = nb_lanes
        self.clock = pygame.time.Clock()
        self.frame = 0
        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption('Traffic simulator')
        self.offset = LANE_W // 2
        self.lanes = self.build_lanes(nb_lanes)
        self.vehicles = None
        self.traffic_rate = 10  # new cars per second
        self.lane_occupancy = None
        self.collision = None

    def build_lanes(self, nb_lanes):
        return tuple(
            {'min': self.offset + n * LANE_W,
             'mid': self.offset + LANE_W / 2 + n * LANE_W,
             'max': self.offset + (n + 1) * LANE_W}
            for n in range(nb_lanes)
        )

    def reset(self):
        # Initialise environment state
        self.frame = 0
        self.vehicles = list()
        self.lane_occupancy = [[] for _ in self.lanes]
        state = list()
        objects = list()
        return state, objects

    def step(self, action):

        free_lanes = set(range(self.nb_lanes))

        for v in self.vehicles:
            v.step()
            lanes_occupied = v.get_lane_set(self.lanes)
            # Remove from the environment cars outside the screen
            if v.position[0] > self.screen_size[0]:
                self.vehicles.remove(v)
                for l in lanes_occupied:
                    self.lane_occupancy[l].remove(v)
            # Check available lanes
            if v.position[0] < v.safe_distance():  # at most safe_distance ahead
                free_lanes -= lanes_occupied

        # Randomly add vehicles
        if random.random() < self.traffic_rate * self.delta_t:
            if free_lanes:
                car = Car([self.lanes[lane] for lane in free_lanes], self.delta_t)
                self.vehicles.append(car)
                for l in car.get_lane_set(self.lanes):
                    self.lane_occupancy[l].append(car)

        # Compute distances
        # distances = list()
        for lane in self.lane_occupancy:
            # distances.append([lane[i].back() - lane[i + 1].front() for i in range(len(lane) - 1)])
            for i in range(1, len(lane)):
                distance = lane[i - 1].back() - lane[i].front()
                safe_distance = lane[i].safe_distance()
                if distance < safe_distance:
                    lane[i].brake(10)
                if distance <= 0:
                    lane[i].colour = colours['r']
                    # Accident, do something!!!
                    self.collision = lane[i]

        # default reward if nothing happens
        reward = -0.001
        done = False
        state = list()

        if self.frame >= 10000:
            done = True

        if done:
            print(f'Episode ended, reward: {reward}, t={self.frame}')

        self.frame += 1

        objects = list()
        return state, reward, done, objects

    def render(self, mode='human'):
        if self.display:

            # self._pause()

            # capture the closing window and mouse-button-up event
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
                elif event.type == pygame.MOUSEBUTTONUP: self._pause()

            # measure time elapsed, enforce it to be >= 1/fps
            self.clock.tick(self.fps)

            # clear the screen
            self.screen.fill(colours['k'])

            # draw lanes
            for lane in self.lanes:
                sw = self.screen_size[0]  # screen width
                draw_dashed_line(self.screen, colours['w'], (0, lane['min']), (sw, lane['min']), 3)
                draw_dashed_line(self.screen, colours['w'], (0, lane['max']), (sw, lane['max']), 3)
                draw_dashed_line(self.screen, colours['r'], (0, lane['mid']), (sw, lane['mid']))

            for v in self.vehicles:
                v.draw(self.screen)

            draw_text(self.screen, f'# cars: {len(self.vehicles)}', (10, 2))

            pygame.display.flip()

            if self.collision: self._pause()

    def _pause(self):
        pause = True
        while pause:
            self.clock.tick(15)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    sys.exit()
                elif e.type == pygame.MOUSEBUTTONUP:
                    pause = False

