#!/bin/bash

for nz in 32; do 
    for dropout in 0.0 0.5; do 
        for layers in 3; do 
            mfile=model=fwd-cnn-ten3-layers=${layers}-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-nhidden=128-fgeom=1-anoise=0.0-zeroact=1-nz=32-beta=0.0-dropout=${dropout}-gclip=5.0-warmstart=0.model
            for sample in fp; do 
                for density in 1.0; do 
                    sbatch submit_eval_fm.slurm $mfile $sample $density
                done
            done
        done
    done
done
    
