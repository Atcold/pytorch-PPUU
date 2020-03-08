#!/bin/bash
#
#SBATCH --job-name=datagen
#SBATCH --output=datagen.out
#SBATCH --error=datagen.err
#SBATCH --time=24:00:00
#SBATCH --gres gpu:0
#SBATCH --exclude="rose5,rose6,vine5,vine11, vine3"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail

source activate py35
cd ../
srun python generate_trajectories.py -time_slot $time_slot -v 1

