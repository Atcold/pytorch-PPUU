#!/bin/bash
#
#SBATCH --job-name=i80-datagen
#SBATCH --time=24:00:00
#SBATCH --gres gpu:0
#SBATCH --exclude="rose5,rose6,vine5,vine11,vine3"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=serkan@nyu.edu

source /data/sk7685/pytorch_11/bin/activate pytorch_11

cd ../
srun python -u generate_trajectories.py -time_slot $1 -map i80
