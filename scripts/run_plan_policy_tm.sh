#!/bin/bash

model_dir=/home/mbhenaff/projects/pytorch-Traffic-Simulator/scratch/models_v7/policy_networks/

for mfile in ${model_dir}/*.model; do 
    sbatch submit_plan.slurm $(basename $mfile) $model_dir 1 1 0 0 policy-tm
done
