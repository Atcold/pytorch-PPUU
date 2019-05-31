#!/bin/bash
#
#SBATCH --job-name=train_fm
#SBATCH --output=logs/train_fm_%j.out
#SBATCH --error=logs/train_fm_%j.err
#SBATCH --time=47:00:00
#SBATCH --gres gpu:1
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11, lion6, lion7"
#SBATCH --constraint="gpu_12gb"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=20000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=serkan@nyu.edu

# module load python-3.6
source /data/sk7685/pytorch_11/bin/activate pytorch_11
cd ../
srun python train_fm.py -model $1 -lrt $2 -nfeature $3 -warmstart $4 -ncond $5 -npred $6 -beta $7 -nz $8 -z_dropout ${9} -layers ${10} -batch_size ${11} -seed ${12} -dropout ${13} 

