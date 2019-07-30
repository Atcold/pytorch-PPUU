#!/bin/bash
#
#SBATCH --job-name=train_fm
##SBATCH --output=logs/train_fm_${15}.out
##SBATCH --error=logs/train_fm_${15}.err
#SBATCH --time=60:00:00
#SBATCH --gres gpu:1
##SBATCH --constraint="gpu_12gb&pascal"
#SBATCH --nodes=1
#SBATCH --mem=50000
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yzs208@nyu.edu

source activate PPUU
cd ../
srun python train_fm.py -model $1 -lrt $2 -nfeature $3 -warmstart $4 -ncond $5 -npred $6 -beta $7 -nz $8 -z_dropout ${9} -layers ${10} -batch_size ${11} -seed ${12} -dropout ${13} -dataset ${14} -data_dir ${15} -model_dir ${16}


