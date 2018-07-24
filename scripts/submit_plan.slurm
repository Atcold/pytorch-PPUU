#!/bin/bash
#
#SBATCH --job-name=plan
#SBATCH --output=plan.out
#SBATCH --error=plan.err
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11, vine6"
#SBATCH --constraint="gpu_12gb&pascal"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=70000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=mbh305@nyu.edu

module load python-3.5
cd ../
srun python plan.py -batch_size 1 -npred $1 -lrt $2 -n_iter 50 -seed 333333 -opt_z $3 -n_rollouts $4 -nexec $5

