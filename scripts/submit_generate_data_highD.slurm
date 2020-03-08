#!/bin/bash
#SBATCH --job-name=highD-datagen
#SBATCH --time=24:00:00
#SBATCH --gres gpu:0
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11 "
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=END,FAIL # notifications for job fail
#SBATCH --mail-user=yzs208@nyu.edu

eval "$(conda shell.bash hook)"
conda activate PPUU
cd ../
srun python -u generate_trajectories.py -map highD -time_slot $time_slot -delta_t 0.04
