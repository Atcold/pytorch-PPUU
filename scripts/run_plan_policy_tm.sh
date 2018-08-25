#!/bin/bash

model_dir=/home/mbhenaff/projects/pytorch-Traffic-Simulator/scratch/models_v9/

for policy_model in ${model_dir}/policy_networks/mbil*.model; do 
    echo $(basename $policy_model)
    sbatch submit_plan_policy_tm.slurm $model_dir $(basename $policy_model)
done
