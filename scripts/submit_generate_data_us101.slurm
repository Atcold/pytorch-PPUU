#!/bin/bash
#
#SBATCH --job-name=us101-datagen
#SBATCH --time=24:00:00
#SBATCH --gres gpu:0
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yzs208@nyu.edu

cd ../
source activate PPUU
srun python -u generate_trajectories.py -time_slot $1 -map us101
source deactivate
