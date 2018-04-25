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
#SBATCH --mail-user=mbh305@nyu.edu

module load python-3.6
cd ../
#srun python eval_fm.py -mfile $1 -batch_size 2 -npred 100 -sampling dist -n_sample 50 -n_batches 100 -save_video 0 -topz_sample 20
srun python eval_fm.py -mfile $1 -batch_size 2 -npred 400 -sampling $2 -n_sample 10 -n_batches 20 -save_video 1 -topz_sample 200

