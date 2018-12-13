# Model-Predictive Policy Learning with Uncertainty Regularization for Driving in Dense Traffic

## Usage

The first step is to train the forward dynamics model (`fm`) on the observational dataset.
This can be done by running:

```
python train_fm.py -model_dir <fm_save_path>
```

Once the dynamics model is trained, it can be used to train the policy network, using `MPUR`, `MPER`, or `IL`.
This is done by running:

```
python train_{MPUR,MPER,IL}.py -model_dir <fm_load_path> -mfile <fm_filename>
```

To evaluate a trained policy, run the script `eval_policy.py`. 

If you are evaluating a `MPUR` model, run:

```
python eval_policy.py -model_dir <load_path> -policy_model_svg <policy_filename> -method policy-svg
```

If you are evaluating a `MPER` model, run:

```
python eval_policy.py -model_dir <load_path> -policy_model_tm <policy_filename> -method policy-tm
```

If you are evaluating an `IL` model, run:

```
python eval_policy.py -model_dir <load_path> -policy_model_il <policy_filename> -method policy-il
```

See the options for details. 
