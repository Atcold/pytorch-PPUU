#!/bin/bash
#
#SBATCH --job-name=train_svg
#SBATCH --output=logs/train_svg_%j.out
#SBATCH --error=logs/train_svg_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --constraint="gpu_12gb&pascal"
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=mbh305@nyu.edu

module load python-3.6
cd ../

srun python train_svg.py -npred $1 -u_reg $2 -lrt_z $3 -z_updates $4 -batch_size $5 -lambda_l $6 -infer_z $7 -lambda_a $8

