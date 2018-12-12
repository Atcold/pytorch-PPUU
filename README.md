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

To evaluate a trained policy, run the script "eval_policy.py". 

If you are evaluating a MPUR model, run:

```
python eval_policy.py -model_dir /where/models/are/saved/ -policy_model_svg /policy/model/file/ -method -policy-svg
```

If you are evaluating a MPER model, run:

```
python eval_policy.py -model_dir /where/models/are/saved/ -policy_model_tm /policy/model/file/ -method policy-tm
```

If you are evaluating an IL model, run:

```
python eval_policy.py -model_dir /where/models/are/saved/ -policy_model_il /policy/model/file/ -method policy-il
```


See the options for details. 

