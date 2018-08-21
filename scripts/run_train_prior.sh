#!/bin/bash 

mfile=model=fwd-cnn-ten3-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-nhidden=128-fgeom=1-zeroact=0-zmult=0-dropout=0.05-nz=32-beta=0.0-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model
for n_mixture in 1 10 50; do 
    for nfeature in 256; do 
        sbatch submit_train_prior.slurm $nfeature $n_mixture $mfile
    done
done
