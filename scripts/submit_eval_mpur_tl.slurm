#!/bin/bash
#
#SBATCH --job-name=eval_MPUR
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --constraint="gpu_12gb&pascal"
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=FAIL # notifications for job done & fail
#SBATCH --mail-user=yzs208@nyu.edu

#eval "$(conda shell.bash hook)"
#conda activate PPUU

source activate PPUU

cd ../
srun python eval_policy_tl.py \
    -seed 333333 \
    -batch_size 1 \
    -method policy-MPUR \
    -policy_model $policy

