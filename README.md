# Documentation goes here

## setup

* create a python3 environment.
* run the following command to install packages
```bash
pip install -r requirements.txt
```

* if you need to run jupyter notebook, run the following command to install all packages:
```bash
pip install -r requirements_full_python_env.txt
```

## development and testing
run `test_planner.py` to replay vectorized data, you can use this to find labeling errors, modeling erros etc.
* add `env.reset(frame=1000, time_interval=1, control=False)` to replay from specific frame and dataset
  * `time_interval`: datasets, collected during different times in a day
  * `frame`: start frame from that datasets

