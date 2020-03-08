#!/bin/bash
#
#SBATCH --job-name=eval_fm
#SBATCH --output=eval_fm.out
#SBATCH --error=eval_fm.err
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11, vine6"
#SBATCH --constraint="gpu_12gb&pascal"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=70000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail

eval "$(conda shell.bash hook)"
conda activate PPUU

cd ../
srun python eval_fm.py -mfile $mfile -batch_size 4 -npred 200 -sampling $sampling -n_sample 10 -n_batches 10 -save_video 1
#srun python eval_fm.py -mfile $1 -batch_size 1 -npred 1000 -sampling $2 -n_sample 10 -n_batches 50 -save_video 1 -graph_density $3 -n_mixture 20
#srun python eval_fm.py -mfile $1 -batch_size 4 -npred 50 -sampling pdf -n_sample 200 -n_batches 100 -save_video 0 -n_mixture 1

