import pygame
import math
import random
import numpy
import sys

from gym import core

seed = 123
random.seed(seed)
numpy.random.seed(seed)

colours = {
    'white': (255, 255, 255),
    'black': (000, 000, 000),
    'red': (255, 000, 000),
    'green': (000, 255, 000),
    'magenta': (255, 000, 255),
    'blue': (000, 000, 255),
}


# class Planet:
#     def __init__(self, x, y, r):
#         self.x = x
#         self.y = y
#         self.r = r
#
#     def draw(self, screen):
#         pygame.draw.circle(screen, white, (int(self.x), int(self.y)), self.r)
#
#
# class Waypoint:
#     def __init__(self, x, y, color_id):
#         self.x = x
#         self.y = y
#         self.r = 10
#         self.colors = [red, green, blue]
#         self.color_id = color_id
#         self.done = 0
#
#     def draw(self, screen):
#         pygame.draw.circle(screen, self.colors[self.color_id], (int(self.x), int(self.y)), int(self.r), 1)
#
#
# class Spaceship:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#         self.r = 20
#         self.dx = 0
#         self.dy = 0
#         self.ux = 0
#         self.uy = 0
#         self.colors = [magenta, red, green, blue]
#
#     def draw(self, action, screen):
#         pygame.draw.circle(screen, self.colors[action], (int(self.x), int(self.y)), int(self.r), 1)
#         pygame.draw.line(screen, red, (int(self.x), int(self.y)), (int(self.x + self.ux*self.r*2), int(self.y + self.uy*self.r*2)))


class StatefulEnv(core.Env):

    lane_width = 50
    offset = 25

    def __init__(self, display=True, dt=4, nb_lanes=4):
        self.display = display
        self.screen_size = (600, (nb_lanes + 1) * self.lane_width)
        self.dt = dt
        self.nb_lanes = nb_lanes
        self.clock = pygame.time.Clock()
        self.t = 0
        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
        self.lanes = self.build_lanes(nb_lanes)

    @classmethod
    def build_lanes(cls, nb_lanes):
        lane_width = cls.lane_width
        offset = cls.offset
        return tuple({'min': offset + n * lane_width,
                      'centre': offset + lane_width / 2 + n * lane_width,
                      'max': offset + (n + 1) * lane_width}
                     for n in range(nb_lanes))

    def reset(self):
        self.t = 0
        state = list()
        objects = list()
        return state, objects

    def step(self, action):

        # default reward if nothing happens
        reward = -0.001
        done = False
        state = list()

        if self.t >= 80:
            done = True

        if done:
            print(f'Episode ended, reward: {reward}, t={self.t}')

        self.t += 1

        objects = list()
        return state, reward, done, objects

    def render(self, mode='human'):
        if self.display:

            # capture the closing window event
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()

            # slow down
            self.clock.tick(90)

            # clear the screen
            self.screen.fill(colours['black'])

            # draw lanes
            for lane in self.lanes:
                pygame.draw.line(self.screen, colours['white'], (0, lane['min']), (self.screen_size[0], lane['min']))
                pygame.draw.line(self.screen, colours['white'], (0, lane['max']), (self.screen_size[0], lane['max']))
                pygame.draw.line(self.screen, colours['red'], (0, lane['centre']), (self.screen_size[0], lane['centre']))

            pygame.display.flip()
