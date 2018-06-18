import numpy as np
from ukfCTRV import ukf
from ukfCTRV import Sensor
import matplotlib.pyplot as plt


sample = 50
x = np.arange(sample)
y = 1.*x + 0.5

plt.plot(x, y)
plt.show()

xnoise = np.random.normal(0, 0.1, x.shape)
x = x + xnoise
ynoise = np.random.normal(0, 1.0, x.shape)
y = y + ynoise

plt.plot(x, y)
plt.show()

np.set_printoptions(linewidth=200, precision=5, suppress=True)
car = ukf.ukf()
sensor = Sensor.I80measurement(name='I80')
sensors = [sensor]

car.sensors = sensors
sensor._ukf = car

px_list = []
py_list = []
for px, py in zip(x,y):
    print(px)
    meas = np.asanyarray([px,py])
    car.process_measurement(meas)
    car.step_filter()
    px_list.append(car.x[0])
    py_list.append(car.x[1])

plt.plot(px_list, py_list)
plt.show()


