from custom_graphics import draw_dashed_line
from traffic_gym import StatefulEnv, Car, colours
import pygame

# Conversion LANE_W from real world to pixels
# A US highway lane width is 3.7 metres, here 50 pixels
LANE_W = 24  # pixels / 3.7 m, lane width
SCALE = LANE_W / 3.7


class RealCar(Car):
    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def safe_distance(self):
        return self._speed * self._safe_factor * 0.5


class RealTraffic(StatefulEnv):
    # Environment's car class
    EnvCar = RealCar

    # Global constants
    SCALE = SCALE
    LANE_W = LANE_W

    def __init__(self, **kwargs):
        kwargs['nb_lanes'] = 6
        kwargs['fps'] = 10
        super().__init__(**kwargs)

        self.screen_size = (67 * self.LANE_W, self.nb_lanes * self.LANE_W + 5 * self.LANE_W)
        # self.photos = (
        #     pygame.image.load('vlcsnap-2018-02-22-14h40m23s503.png'),
        #     pygame.image.load('vlcsnap-2018-02-23-10h55m01s517.png'),
        #     pygame.image.load('vlcsnap-2018-03-08-16h22m49s299.png')
        # )
        # self.photos_rect = (
        #     self.photos[0].get_rect().move([0, 22]),
        #     self.photos[1].get_rect().move([928, 22 + 4]),
        #     self.photos[2].get_rect().move([1258, 22 + 5])
        # )
        if self.display:  # if display is required
            self.screen = pygame.display.set_mode(self.screen_size)  # set screen size
        # self.delta_t = 1 / 10  # simulation timing interval

    def _draw_lanes(self, mode='human'):
        if mode == 'human':
            lanes = self.lanes
            s = self.screen
            draw_line = pygame.draw.line
            w = colours['w']

            for lane in self.lanes:
                sw = self.screen_size[0]  # screen width
                draw_dashed_line(s, w, (0, lane['min']), (sw, lane['min']), 3)
                draw_dashed_line(s, colours['r'], (0, lane['mid']), (sw, lane['mid']))

            draw_line(s, w, (0, lanes[0]['min']), (sw, lanes[0]['min']), 3)
            bottom = lanes[-1]['max']
            draw_line(s, w, (0, bottom), (18 * LANE_W, bottom), 3)
            draw_line(s, w, (0, bottom + 29), (18 * LANE_W, bottom + 29 - 0.035 * 18 * LANE_W), 3)
            draw_dashed_line(s, w, (18 * LANE_W, bottom + 13), (31 * LANE_W, bottom), 3)
            sw *= .9
            draw_dashed_line(s, colours['r'], (0, bottom + 42), (sw, bottom + 42 - 0.035 * sw))
            draw_line(s, w, (0, bottom + 53), (sw, bottom + 53 - 0.035 * sw), 3)
            draw_line(s, w, (sw, bottom + 3), (self.screen_size[0], bottom + 2), 3)
