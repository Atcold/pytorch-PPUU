#!/bin/bash

for sample in fp; do 
    for density in 1.0; do 
        sbatch submit_eval_fm.slurm model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1-nz=32-beta=0.0-nmix=1-warmstart=1.model $sample $density
    done
done

for sample in knn; do 
    for density in 0.001 0.002 0.005; do 
        sbatch submit_eval_fm.slurm model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1-nz=32-beta=0.0-nmix=1-warmstart=1.model $sample $density
    done
done

sbatch submit_eval_fm.slurm model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1-nz=32-beta=0.0-nmix=1-warmstart=1.model pdf 1.0
    
