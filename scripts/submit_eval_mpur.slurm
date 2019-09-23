#!/bin/bash
#
#SBATCH --job-name=eval_MPUR
#SBATCH --output=logs_eval_mpur/eval_MPUR_%j.out
#SBATCH --error=logs_eval_mpur/eval_MPUR_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --constraint="gpu_12gb&pascal"
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11, lion1, lion2, lion3"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=40000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=serkan@nyu.edu

eval "$(conda shell.bash hook)"
conda activate PPUU

cd ../
srun python eval_policy.py \
    -method policy-MPUR \
    -policy_model $policy \
    -sftmx_beta $sftmx_beta

