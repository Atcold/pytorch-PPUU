#!/bin/bash 

for loss in pdf; do 
    for u_sphere in 0; do 
        for nfeature in 96; do 
            sbatch submit_train_prior.slurm $nfeature $loss $u_sphere
        done
    done
done
