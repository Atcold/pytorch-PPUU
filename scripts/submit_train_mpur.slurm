#!/bin/bash
#
#SBATCH --job-name=train_MPUR
#SBATCH --output=logs_mpur/train_MPUR_%j.out
#SBATCH --error=logs_mpur/train_MPUR_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --constraint="gpu_12gb&pascal"
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11, lion1, lion2, lion3"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=40000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=serkan@nyu.edu

eval "$(conda shell.bash hook)"
conda activate PPUU

cd ../
srun python train_MPUR.py \
    -npred $npred \
    -u_reg $u_reg \
    -lrt_z $lrt_z \
    -z_updates $z_updates \
    -batch_size $batch_size \
    -lambda_l $lambda_l \
    -lambda_a $lambda_a \
    -seed $seed \
    -sftmx_beta $sftmx_beta \
    -policy 'policy-deterministic'

