#!/bin/bash
#
#SBATCH --job-name=train_fm
#SBATCH --output=train_fm.out
#SBATCH --error=train_fm.err
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --exclude="rose5,rose6,vine5,vine11"
#SBATCH --constraint="gpu_12gb"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=50000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=mbh305@nyu.edu

module load python-3.6
cd ../
srun python train_fm.py -model $1 -lrt $2 -nfeature $3 -n_hidden $4 -ncond $5 -npred $6 -beta $7 -nz $8

