#!/bin/bash
#
#SBATCH --job-name=train_mb_il
#SBATCH --output=logs/train_mb_il_%j.out
#SBATCH --error=logs/train_mb_il_%j.err
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

srun python train_mb_il.py -nfeature $1 -n_hidden $2 -seed $3 -npred $4 -lambda_c $5 -targetprop $6 -gamma $7 -batch_size $8 -curriculum_length $9 -actions_subsample ${10} -lambda_h ${11} -context_dim ${12} 

