#!/bin/bash
#
#SBATCH --job-name=train_fm_det
#SBATCH --output=logs/train_fm_%j.out
#SBATCH --error=logs/train_fm_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11, lion6, lion7"
#SBATCH --constraint="gpu_12gb&pascal"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=50000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=canziani@nyu.edu

# module load python-3.6
cd ../
srun python train_fm.py -model $1 -lrt $2 -nfeature $3 -warmstart $4 -ncond $5 -npred $6 -beta $7 -nz $8 -z_dropout ${9} -layers ${10} -batch_size ${11} -seed ${12} -dropout ${13}

