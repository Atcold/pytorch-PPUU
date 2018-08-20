## Getting started

### Installation
* `pip install pygame`
* MacOs : Install XQuartz <https://www.xquartz.org/>


### Description
The action space is R^2
* The first element is the acceleration
* The second element is the norm of the direction gradient

Car class :
* Car state is defined by x, y, dx, dy
* Distance function? 

There are 2 types of cars :
* uncontrolled : Their policy is based on the ground truth.
* controlled : The action is the one passed through the step function. One can define one's own policy, generate an action and pass it to the step function.

### Samples
* `python play_maps` will play a scene with all cars being uncontrolled, you are visualizing the ground truth
* `python test_planner` will play a scene with all cars being uncontrolled but one. One car is controlled based on the trivial policy of going straight, 0 acceleration, 0 steering.
