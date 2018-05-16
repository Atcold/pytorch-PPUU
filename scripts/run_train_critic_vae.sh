#!/bin/bash

mfile=model=fwd-cnn-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-nz=32-beta=0.0-warmstart=0.model
for sampling in fp; do 
    for n_samples in 1; do 
        for seed in 4; do 
            sbatch submit_train_critic.slurm $mfile $sampling $seed 1
        done
    done
done


mfile=model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=1e-05-warmstart=1.model
for sampling in fp; do 
    for n_samples in 1; do 
        for seed in 4; do 
            sbatch submit_train_critic.slurm $mfile $sampling $seed 1
        done
    done
done


mfile=model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=0.0001-warmstart=1.model
for sampling in fp; do 
    for n_samples in 1; do 
        for seed in 4; do 
            sbatch submit_train_critic.slurm $mfile $sampling $seed 1
        done
    done
done


