#!/bin/bash
#
#SBATCH --job-name=train_policy_prior
#SBATCH --output=logs/train_policy_prior_%j.out
#SBATCH --error=logs/train_policy_prior_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --constraint="gpu_12gb&pascal"
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail

module load python-3.6
cd ../

srun python train_policy_prior.py -mfile $1

