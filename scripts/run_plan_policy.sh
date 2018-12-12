#!/bin/bash

model_dir=/home/mbhenaff/projects/pytorch-Traffic-Simulator/scratch/models_v11/

method=policy-tm

#for policy_model in ${model_dir}/policy_networks/*svg*vae*learnedcost=*.model; do 
#for policy_model in ${model_dir}/policy_networks/*policy-il*lrt=0.0001*nmixture=1-*.model; 
#for policy_model in ${model_dir}/policy_networks/*svg*zdropout=0.5*npred=30*ureg=0.05*inferz=0*learnedcost=1*seed=3*novalue.model; do 
#for policy_model in ${model_dir}/policy_networks/*svg*deterministic*npred=40*novalue.model; do 
#for policy_model in ${model_dir}/policy_networks/*svg*zdropout=0.5*npred=30*depeweg=1*novalue.model; do 
#for policy_model in ${model_dir}/policy_networks/*svg*npred=*lambdaa=0*5*gamma*.model; do
#for policy_model in ${model_dir}/policy_networks/*svg*npred=3-*zdropout=0.0*seed=2*novalue.model; do
for seed in 2 3; do 
    for policy_model in ${model_dir}/policy_networks/*mbil*npred=5-*seed=${seed}*.model; do
#for policy_model in ${model_dir}/policy_networks/*svg*deterministic*npred=3-*lambdaa=0.0-*seed=2*novalue.model; do
        echo $(basename $policy_model)
        sbatch submit_plan_policy.slurm $method $model_dir $(basename $policy_model)
    done
done
