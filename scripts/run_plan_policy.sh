#!/bin/bash

model_dir=/home/mbhenaff/projects/pytorch-Traffic-Simulator/scratch/models_v11/

method=policy-svg

#for policy_model in ${model_dir}/policy_networks/*svg*vae*learnedcost=*.model; do 
#for policy_model in ${model_dir}/policy_networks/*policy-il*lrt=0.0001*nmixture=1-*.model; 
for policy_model in ${model_dir}/policy_networks/*svg*npred=20-*lambdaa=0.01*.model; do 
    echo $(basename $policy_model)
    sbatch submit_plan_policy.slurm $method $model_dir $(basename $policy_model)
done
