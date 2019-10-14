#!/bin/bash
#
#SBATCH --job-name=plan
#SBATCH --output=logs/plan_%j.out
#SBATCH --error=logs/plan_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11, vine6"
#SBATCH --constraint="gpu_12gb&pascal"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=70000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail

cd ../
srun python plan_with_uncertainty.py -method $1 -bprop_niter $2 -bprop_lrt $3 -u_reg $4 -u_hinge $5 -bprop_buffer $6 -lambda_l $7

