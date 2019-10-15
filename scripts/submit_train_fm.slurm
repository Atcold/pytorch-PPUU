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

eval "$(conda shell.bash hook)"
conda activate PPUU

cd ../
srun python train_fm.py \
    -model $model \
    -lrt $lrt \
    -nfeature $nfeature \
    -warmstart $warmstart \
    -ncond $ncond \
    -npred $npred \
    -beta $beta \
    -nz $nz \
    -z_dropout $z_dropout \
    -layers $layers \
    -batch_size $batch_size \
    -seed $seed \
    -dropout $dropout

