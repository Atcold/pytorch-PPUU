#!/bin/bash
#
#SBATCH --job-name=eval_MPUR
#SBATCH --output=logs/eval_MPUR_%j.out
#SBATCH --error=logs/eval_MPUR_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --constraint="gpu_12gb&pascal"
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11"
#SBATCH --cpus-per-task=7
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail

eval "$(conda shell.bash hook)"
conda activate PPUU

cd ../
srun python eval_policy.py \
    $@ \
    -model_dir $model_dir \
    -method policy-MPUR \
    -policy_model $policy \
    -save_grad_vid
