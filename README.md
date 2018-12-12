# Model-Predictive Policy Learning with Uncertainty Regularization for Driving in Dense Traffic

## Usage

The first step is to train the forward dynamics model on the observational dataset. This can be done by running:

```
python train_fm.py -model_dir /where/to/save/model/
```

Once the dynamics model is trained, it can be used to train the policy network. This is done by running:

```
python train_policy_net.py -model_dir /where/dynamics/model/is/saved/ -mfile /dynamics/model/file/
```