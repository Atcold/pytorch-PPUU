#!/bin/bash

for seed in 16 17 18 19 20; do 
    sbatch submit_train_critic_joint.slurm $seed
done
