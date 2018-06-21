# Unscented Kalman Filter with Bicycle motion model

This model implements an online version Ukf with a Bicycle motion model. The Ukf can track vehicles with a state vector `[px, py, speed, hdg (rad)]`, i.e. location (x, y), speed, and heading angle, and update the state vector with measurement (observation) of positions `(px, py)`. 

# How to use it.
```python
from ukfBicycle import ukf

# create a ukf object
car = ukf(dt=0.1, wheelbase=2.5, startx=0., starty=0., startspeed=0., starthdg=0., stdx=1.0, stdy=1.0, noise=0.02)
```
* `stdx` and `stdy` describe noise of measurement
* `noise` describes process noise

For tracking applications, when a new measurement `z=np.array([px, py])` is ready, update the ukf:
```python
car.update(z)
```

The state vector can be accessed:
```python
px = car.ukf.x[0]
py = car.ukf.x[1]
speed = car.ukf.x[2]
hdg = car.ukf.x[3]
```

For control applications, action `u=np.array([acceleration, steering_angle])` can be sent with measurement together to update the Ukf:
```python
car.step(z, action=u)
```
