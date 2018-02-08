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
LANE_W = 50  # pixels / 3.7 m, lane width
SCALE = LANE_W / 3.7

colours = {
    'w': (255, 255, 255),
    'k': (000, 000, 000),
    'r': (255, 000, 000),
    'g': (000, 255, 000),
    'm': (255, 000, 255),
    'b': (000, 000, 255),
    'c': (000, 255, 255),
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
        self.speed = 100 * 1000 / 3600 * SCALE  # m / s
        self.dt = dt

    def draw(self, screen, colour):
        """
        Draw current car on screen with a specific colour
        :param screen: PyGame ``Surface`` where to draw
        :param colour: in a (R, G, B) format
        """
        x, y = self.position
        rectangle = (int(x), int(y), self.length, self.width)
        pygame.draw.rect(screen, colour, rectangle)

    def step(self):  # takes also the parameter action = state temporal derivative
        """
        Update current position, given current velocity and acceleration
        """
        self.position += self.speed * self.direction * self.dt

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


class StatefulEnv(core.Env):

    def __init__(self, display=True, nb_lanes=4, fps=30):
        self.display = display
        self.screen_size = (20 * LANE_W, (nb_lanes + 1) * LANE_W)
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
        self.vehicles = list()
        self.traffic_rate = 2  # new cars per second

    def build_lanes(self, nb_lanes):
        return tuple(
            {'min': self.offset + n * LANE_W,
             'mid': self.offset + LANE_W / 2 + n * LANE_W,
             'max': self.offset + (n + 1) * LANE_W}
            for n in range(nb_lanes)
        )

    def reset(self):
        self.frame = 0
        state = list()
        objects = list()
        self.vehicles = list()
        return state, objects

    def step(self, action):

        free_lanes = set(range(self.nb_lanes))

        for v in self.vehicles:
            v.step()
            # Remove from the environment cars outside the screen
            if v.position[0] > self.screen_size[0]:
                self.vehicles.remove(v)
            # Check available lanes
            if v.position[0] < 0 + v.length:  # v.length as safety distance for the moment
                free_lanes -= v.get_lane_set(self.lanes)

        # Randomly add vehicles
        if random.random() < self.traffic_rate * self.delta_t:
            self.vehicles.append(Car([self.lanes[lane] for lane in free_lanes], self.delta_t))

        # default reward if nothing happens
        reward = -0.001
        done = False
        state = list()

        if self.frame >= 800:
            done = True

        if done:
            print(f'Episode ended, reward: {reward}, t={self.frame}')

        self.frame += 1

        objects = list()
        return state, reward, done, objects

    def render(self, mode='human'):
        if self.display:

            # capture the closing window event
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()

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
                v.draw(self.screen, colours['c'])

            draw_text(self.screen, f'# cars: {len(self.vehicles)}', (10, 2))

            pygame.display.flip()
