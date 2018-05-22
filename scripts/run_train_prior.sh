#!/bin/bash 

for n_mixture in 20 50 100 200 500; do 
    for nfeature in 128; do 
        sbatch submit_train_prior.slurm $nfeature $n_mixture
    done
done
