import pygame
import math
import random
import numpy as np
import sys
from custom_graphics import draw_dashed_line

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
    def __init__(self, nb_lanes, offset, fps):
        """
        Initialise a sedan on a random lane
        :param nb_lanes:
        """
        self.length = round(4.8 * SCALE)
        self.width = round(1.8 * SCALE)
        self.direction = np.array((1, 0), np.float)
        self.position = np.array((
            -self.length,
            offset + (random.randrange(nb_lanes) + 0.5) * LANE_W - self.width // 2
        ), np.float)
        self.speed = 100 * 1000 / 3600 * SCALE  # m / s
        self.fps = fps

    def draw(self, screen, colour):
        x, y = self.position
        rectangle = (int(x), int(y), self.length, self.width)
        pygame.draw.rect(screen, colour, rectangle)

    def step(self):
        self.position += self.speed * self.direction / self.fps


class StatefulEnv(core.Env):

    def __init__(self, display=True, nb_lanes=4, fps=30, dt=None):
        self.display = display
        self.screen_size = (20 * LANE_W, (nb_lanes + 1) * LANE_W)
        self.fps = fps
        self.dt = dt
        self.nb_lanes = nb_lanes
        self.clock = pygame.time.Clock()
        self.frame = 0
        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
        self.offset = LANE_W // 2
        self.lanes = self.build_lanes(nb_lanes)
        self.vehicles = list()

    def build_lanes(self, nb_lanes):
        return tuple(
            {'min': self.offset + n * LANE_W,
             'centre': self.offset + LANE_W / 2 + n * LANE_W,
             'max': self.offset + (n + 1) * LANE_W}
            for n in range(nb_lanes)
        )

    def reset(self):
        self.frame = 0
        state = list()
        objects = list()
        self.vehicles = [Car(self.nb_lanes, self.offset, self.fps)]
        return state, objects

    def step(self, action):

        for v in self.vehicles:
            v.step()

        # default reward if nothing happens
        reward = -0.001
        done = False
        state = list()

        if self.frame >= 80:
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
                screen_width = self.screen_size[0]
                draw_dashed_line(self.screen, colours['w'], (0, lane['min']), (screen_width, lane['min']), 3)
                draw_dashed_line(self.screen, colours['w'], (0, lane['max']), (screen_width, lane['max']), 3)
                draw_dashed_line(self.screen, colours['r'], (0, lane['centre']), (screen_width, lane['centre']))

            for v in self.vehicles:
                v.draw(self.screen, colours['c'])

            pygame.display.flip()
