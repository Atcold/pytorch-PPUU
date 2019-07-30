#!/bin/bash
#
#SBATCH --job-name=highD-datagen
#SBATCH --time=24:00:00
#SBATCH --gres gpu:0
#SBATCH --exclude="rose5,rose6,vine5,vine11,vine3"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=FAIL # notifications for job fail
#SBATCH --mail-user=yzs208@nyu.edu

source activate PPUU
cd ../
srun python -u generate_trajectories.py -map highD -recording $1 -delta_t 0.04
