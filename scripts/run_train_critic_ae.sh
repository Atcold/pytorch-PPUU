#!/bin/bash


mfile=model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1-nz=32-beta=0.0-nmix=1-warmstart=1.model

for sampling in fp; do 
    for density in 1.0; do 
        for seed in 4 5; do 
            sbatch submit_train_critic.slurm $mfile $sampling $seed $density
        done
    done
done
