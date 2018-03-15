#!/bin/bash
#
#SBATCH --job-name=train_il
#SBATCH --output=train_il.out
#SBATCH --error=train_il.err
#SBATCH --time=24:00:00
#SBATCH --gres gpu:1
#SBATCH --constraint="gpu_12gb"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=mbh305@nyu.edu

module load python-3.6
cd ../
srun python train_il.py -model $1 -lrt $2 -n_hidden $3 -ncond $4 -npred $5 -beta $6 -nshards $7 -nz $8

