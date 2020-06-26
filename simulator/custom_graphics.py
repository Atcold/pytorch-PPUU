import pygame
import math
import numpy as np


class Point:
    # constructed using a normal tupple
    def __init__(self, point_t=(0, 0)):
        self.x = float(point_t[0])
        self.y = float(point_t[1])

    # define all useful operators
    def __add__(self, other):
        return Point((self.x + other.x, self.y + other.y))

    def __sub__(self, other):
        return Point((self.x - other.x, self.y - other.y))

    def __mul__(self, scalar):
        return Point((self.x * scalar, self.y * scalar))

    def __truediv__(self, scalar):
        return Point((self.x / scalar, self.y / scalar))

    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    # get back values in original tuple format
    def get(self):
        return self.x, self.y


def draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=10):
    origin = Point(start_pos)
    target = Point(end_pos)
    displacement = target - origin
    length = displacement.norm()
    slope = displacement / length

    for index in range(0, round(length / dash_length), 2):
        start = origin + (slope * index * dash_length)
        end = origin + (slope * (index + 1) * dash_length)
        pygame.draw.line(surf, color, start.get(), end.get(), width)


def draw_text(screen, text, xy, font_size=30, colour=(255, 255, 255), font=None):
    if font is None:
        font = pygame.font.SysFont(None, font_size)
    text = font.render(text, True, colour)
    text_rect = text.get_rect()
    text_rect.left = xy[0]
    text_rect.top = xy[1]
    screen.blit(text, text_rect)


def draw_rect(screen, colour, rect, direction=(1, 0), thickness=0):
    x, y, l, w = rect
    xy = np.array(((x, y - w/2), (x, y + w/2), (x + l, y + w/2), (x + l, y - w/2)))
    c, s = direction
    rot = np.array(((c, -s), (s, c)))
    xy = (rot @ (xy - (x, y)).T).T + (x, y)
    return pygame.draw.polygon(screen, colour, xy, thickness)
