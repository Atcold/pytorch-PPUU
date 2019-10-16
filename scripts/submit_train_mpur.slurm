#!/bin/bash
#
#SBATCH --job-name=train_MPUR
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --constraint="gpu_12gb&pascal"
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=yzs208@nyu.edu

#eval "$(conda shell.bash hook)"
#conda activate PPUU
source activate PPUU

cd ../
srun python train_MPUR.py \
    -model_dir $model_dir \
    -learned_cost \
    -npred $npred \
    -u_reg $u_reg \
    -batch_size $batch_size \
    -lambda_l $lambda_l \
    -lambda_a $lambda_a \
    -lambda_tl $lambda_tl \
    -seed $seed \
    -policy 'policy-deterministic'

