#!/bin/bash
#
#SBATCH --job-name=train_critic
#SBATCH --output=train_critic.out
#SBATCH --error=train_critic.err
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11, vine6"
#SBATCH --constraint="gpu_12gb&pascal"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=70000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail

module load python-3.6
cd ../
srun python train_critic_joint.py -mfile $1 -batch_size 2 -npred 200 -lrt 0.0001 -sampling $2 -seed $3 -graph_density $4 -n_mixture $5
