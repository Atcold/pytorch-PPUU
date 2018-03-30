#!/bin/bash 

for loss in sphere; do 
    for u_sphere in 1; do 
        for nfeature in 128 256; do 
            sbatch submit_train_prior.slurm $nfeature $loss $u_sphere
        done
    done
done
