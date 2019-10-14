#!/bin/bash
#
#SBATCH --job-name=train_value
#SBATCH --output=train_value.out
#SBATCH --error=train_value.err
#SBATCH --time=24:00:00
#SBATCH --gres gpu:1
#SBATCH --constraint="gpu_12gb&pascal"
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail

module load python-3.6
cd ../
srun python train_value_function.py -npred $1 -dropout $2 -gamma $3 -nsync $4 -nfeature $5 -n_hidden $5

