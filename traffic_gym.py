import pygame
import math
import random
import numpy
import sys

from gym.envs.registration import register
from gym import core

seed = 123
random.seed(seed)
numpy.random.seed(seed)

register(
    id='Traffic-v0',
    entry_point='traffic_gym:StatefulEnv',
    tags={'wrapper_config.TimeLimit.max_episodesteps': 100},
)

white   = (255, 255, 255)
black   = (000, 000, 000)
red     = (255, 000, 000)
green   = (000, 255, 000)
magenta = (255, 000, 255)
blue    = (000, 000, 255)


class Planet:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def draw(self, screen):
        pygame.draw.circle(screen, white, (int(self.x), int(self.y)), self.r)


class Waypoint:
    def __init__(self, x, y, color_id):
        self.x = x
        self.y = y
        self.r = 10
        self.colors = [red, green, blue]
        self.color_id = color_id
        self.done = 0

    def draw(self, screen):
        pygame.draw.circle(screen, self.colors[self.color_id], (int(self.x), int(self.y)), int(self.r), 1)


class Spaceship:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.r = 20
        self.dx = 0
        self.dy = 0
        self.ux = 0
        self.uy = 0
        self.colors = [magenta, red, green, blue]

    def draw(self, action, screen):
        pygame.draw.circle(screen, self.colors[action], (int(self.x), int(self.y)), int(self.r), 1)
        pygame.draw.line(screen, red, (int(self.x), int(self.y)), (int(self.x + self.ux*self.r*2), int(self.y + self.uy*self.r*2)))


class StatefulEnv(core.Env):
    def __init__(self):
        pass

    def setup(self, screen_size=600, display=True, dt=4, thrust=0.1, G=0.015, n_planets=3):
        self.display = display
        self.screen_size = screen_size
        self.thrust = thrust
        self.G = G
        self.dt = dt
        self.n_planets = n_planets

        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_size, screen_size))

        self.clock = pygame.time.Clock()
        self.circle = pygame.Surface((30, 30,))
        self.reset()

    def reset(self):
        self.running = True
        self.t = 0
        self.stars = [(random.randint(0, self.screen_size - 1), random.randint(0, self.screen_size - 1)) for x in range(140)]
        self.planets = []
        if self.n_planets > 0:
            self.planets.append(Planet(random.randint(50, self.screen_size - 50), random.randint(50, self.screen_size - 50), random.randint(10, 40)))
            while len(self.planets) < self.n_planets:
                new_planet = Planet(random.randint(50, self.screen_size - 50), random.randint(50, self.screen_size - 50), random.randint(10, 40))
                ok = True
                for p in self.planets:
                    if self._detect_collision(p, new_planet):
                        ok = False
                if ok:
                    self.planets.append(new_planet)

        self.waypoints = []
        while len(self.waypoints) < 3:
            new_waypoint = Waypoint(random.randint(50, self.screen_size - 50), random.randint(50, self.screen_size - 50), len(self.waypoints))
            ok = True
            for p in self.planets:
                if self._detect_collision(p, new_waypoint):
                    ok = False
            if ok:
                self.waypoints.append(new_waypoint)

        ok = False
        self.ship = None
        while not ok:
            self.ship = Spaceship(random.randint(50, self.screen_size - 50), random.randint(50, self.screen_size - 50))
            collision = False
            for p in self.planets:
                collision = collision or self._detect_collision(p, self.ship)
            for w in self.waypoints:
                collision = collision or self._detect_collision(w, self.ship)
            ok = not collision

        return self._collect_state(), [self.ship, self.planets, self.waypoints]

    def _collect_state(self):
        s = [self.ship.x, self.ship.y, self.ship.dx, self.ship.dy]
        for p in self.planets:
            s += [p.x, p.y, p.r]
        for w in self.waypoints:
            s += [w.x, w.y, w.r]
        s = [float(i) for i in s]
        return s

    def _collect_action(self, ux, uy, discrete_action):
        one_hot = numpy.zeros(4)
        one_hot[discrete_action] = 1
        a = numpy.concatenate((numpy.array((ux, uy)), one_hot))
        return a

    def _detect_collision(self, object1, object2):
        dx = object1.x - object2.x
        dy = object1.y - object2.y
        distance = math.hypot(dx, dy)
        if distance < (object1.r + object2.r):
            return True
        return False

    def step(self, action):
        ux = action[0]
        uy = action[1]
        discrete_action = action[2:]
        self.discrete_action = discrete_action.argmax()

        # default reward if nothing happens
        reward = -0.001
        done = False

        # check if ship is crashed
        crashed = False
        for p in self.planets:
            if self._detect_collision(p, self.ship):
                crashed = True
                reward = -1
                print('crashed')
                exit_condition = 'crash'

        # only update if ship is not crashed
        if not crashed:
            # check if it reaches a waypoint
            reached_waypoint = False
            for k in range(len(self.waypoints)):
                if self._detect_collision(self.waypoints[k], self.ship):
                    reached_waypoint = True
                    if self.discrete_action == k + 1 and self.waypoints[k].done == 0:
                        # reached this waypoint
                        print('success :)')
                        reward = 1
                        exit_condition = 'success'
                    elif self.discrete_action != k + 1 and self.discrete_action != 0:
                        # dropped the wrong cargo
                        print('delivered wrong cargo')
                        reward = -1
                        exit_condition = 'wrong_cargo'

            # check if it dropped cargo in space
            if not reached_waypoint:
                if self.discrete_action != 0:
                    print('dropped cargo')
                    reward = -1
                    exit_condition = 'dropped'

            # threshold thrust
            norm = math.sqrt(ux**2 + uy**2)
            u_clip = 0.15
            if norm > u_clip:
                ux /= (norm/u_clip)
                uy /= (norm/u_clip)

            self.ship.ux = ux
            self.ship.uy = uy
            # compute acceleration
            a_x, a_y = 0, 0
            # apply control
            a_x += ux
            a_y += uy
            # add gravity
            for p in self.planets:
                rx = p.x - self.ship.x
                ry = p.y - self.ship.y
                dist = math.sqrt((p.x-self.ship.x)**2 + (p.y-self.ship.y)**2)
                a_x += self.G*rx * self.ship.r*p.r/(dist**2)
                a_y += self.G*ry * self.ship.r*p.r/(dist**2)

            # add damping constant
            a_x -= 0.1 * self.ship.dx
            a_y -= 0.1 * self.ship.dy

            # update velocity
            self.ship.dx += self.dt*a_x
            self.ship.dy += self.dt*a_y

            # update position
            self.ship.x += self.dt*self.ship.dx
            self.ship.y += self.dt*self.ship.dy

        state = self._collect_state()

        if self.display:
            self.render()

        if self.t >= 80:
            done = True
            exit_condition = 'timeout'

        if done:
            print('Episode ended, reward: {}, t={}'.format(reward, self.t))

        self.clock.tick(90)
        self.t += 1

        return state, reward, done, [self.ship, self.planets, self.waypoints]

    def render(self, mode='human'):
        if self.display:# capture the closing window event
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
            # draw the stars
            self.screen.fill(black)
            for star in self.stars:
                star_x, star_y = star[0], star[1]
                pygame.draw.line(self.screen, white, (star_x, star_y), (star_x, star_y))
            # draw planets
            for p in self.planets:
                p.draw(self.screen)
            # draw waypoints
            for w in self.waypoints:
                w.draw(self.screen)
            self.ship.draw(self.discrete_action, self.screen)
            pygame.display.flip()