# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from matplotlib.pylab import *

# %%
style.use(['dark_background', 'bmh'])
rc('axes', facecolor='k')
rc('figure', facecolor='k')
rc('figure', figsize=(16, 4))

# %%
screen_length = 117
ego_position = 58

front_position = 80
back_position = 20
safe_distance = 10

target_speed = 20
current_speed = 30

max_slope = 0.2 / safe_distance

# %%
x = r_[0:screen_length:1]

front_cost = maximum(1 - abs(front_position - x) / safe_distance, 0)
back_cost = maximum(1 - abs(back_position - x) / safe_distance, 0)

slope = (current_speed - target_speed) / 100
slope = maximum(minimum(slope, max_slope), -max_slope)
speed_cost = slope * x + abs(min(slope * x))

# %%
# Color shorthands
r, g, b, p = 'C1', 'C3', 'C0', 'C2'
set_color = lambda c: dict(linefmt=c, basefmt=" ", markerfmt='o'+c)

# %%
figure()
stem(x, front_cost + back_cost + speed_cost, **set_color(p), label='Total Cost')
# stem(x, front_cost, **set_color(r), label='Front Cost')
# stem(x, back_cost, **set_color(b), label='Back Cost')
stem(x, speed_cost, **set_color(g), label='Speed Cost')
arrow_props = dict(width=1.5, facecolor='white')
# annotate('Ego Car', (ego_position, 0.0), (ego_position, -0.5), arrowprops=arrow_props)
annotate('Front Car', (front_position, 0.0), (front_position, -0.5), arrowprops=arrow_props)
annotate('Back Car', (back_position, 0.0), (back_position, -0.5), arrowprops=arrow_props)
axis('equal')
title('Speed Cost + Proximity Cost')
legend()
savefig('speed_cost.png')

# %%
figure()
stem(x[0:40], (front_cost + back_cost + speed_cost)[0:40], **set_color(p), label='Total Cost')
# stem(x, front_cost, **set_color(r), label='Front Cost')
# stem(x, back_cost, **set_color(b), label='Back Cost')
stem(x[0:40], speed_cost[0:40], **set_color(g), label='Speed Cost')
arrow_props = dict(width=1.5, facecolor='white')
# annotate('Ego Car', (ego_position, 0.0), (ego_position, -0.5), arrowprops=arrow_props)
# annotate('Front Car', (front_position, 0.0), (front_position, -0.5), arrowprops=arrow_props)
annotate('Back Car', (back_position, 0.0), (back_position, -0.5), arrowprops=arrow_props)
axis('equal')
title('Speed Cost + Proximity Cost')
legend()
savefig('back_car_cost.png')

# %%
figure()
stem(x[60:100], (front_cost + back_cost + speed_cost)[60:100], **set_color(p), label='Total Cost')
# stem(x, front_cost, **set_color(r), label='Front Cost')
# stem(x, back_cost, **set_color(b), label='Back Cost')
stem(x[60:100], speed_cost[60:100], **set_color(g), label='Speed Cost')
arrow_props = dict(width=1.5, facecolor='white')
# annotate('Ego Car', (ego_position, 0.0), (ego_position, -0.5), arrowprops=arrow_props)
annotate('Front Car', (front_position, 0.0), (front_position, -0.5), arrowprops=arrow_props)
# annotate('Back Car', (back_position, 0.0), (back_position, -0.5), arrowprops=arrow_props)
axis('equal')
title('Speed Cost + Proximity Cost')
legend()
savefig('front_car_cost.png')

# %%
