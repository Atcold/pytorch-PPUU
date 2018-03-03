#!/bin/bash
#
#SBATCH --job-name=datagen
#SBATCH --output=datagen.out
#SBATCH --error=datagen.err
#SBATCH --time=24:00:00
#SBATCH --gres gpu:0
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=mbh305@nyu.edu

module load python-3.6
source activate py36
cd ../
srun python generate_trajectories.py -n_episodes 50 -seed $1

