#!/bin/bash
#
#SBATCH --job-name=datagen
#SBATCH --time=24:00:00
#SBATCH --gres gpu:0
#SBATCH --exclude="rose5,rose6,vine5,vine11,vine3"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=canziani@nyu.edu

cd ../
srun python -u generate_trajectories.py -time_slot $1 -map lanker

