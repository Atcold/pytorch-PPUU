#!/bin/bash
#
#SBATCH --job-name=train_il
#SBATCH --output=train_il.out
#SBATCH --error=train_il.err
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
srun python train_il.py -model $1 -lrt $2 -n_hidden $3 -ncond $4 -npred $5 -beta $6 -nz $7 -n_mixture $8 -seed $9

