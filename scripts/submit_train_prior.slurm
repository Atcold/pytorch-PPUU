#!/bin/bash
#
#SBATCH --job-name=train_prior
#SBATCH --output=train_prior.out
#SBATCH --error=train_prior.err
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11"
#SBATCH --constraint="gpu_12gb&pascal"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=50000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail

module load python-3.6
cd ../
srun python train_prior.py -nfeature $1 -n_mixture $2 -mfile $3

