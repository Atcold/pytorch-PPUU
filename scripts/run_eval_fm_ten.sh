#!/bin/bash

for dropout in 0.0 0.25 0.5; do 
    mfile=model=fwd-cnn-ten-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-combine=add-nz=32-beta=0.0-dropout=${dropout}-gclip=1.0-warmstart=1.model
    for sample in fp; do 
        for density in 1.0; do 
            sbatch submit_eval_fm.slurm $mfile $sample $density
        done
    done

#    for sample in knn; do 
#        for density in 0.005; do 
#            sbatch submit_eval_fm.slurm $mfile $sample $density
#        done
#    done

    #sbatch submit_eval_fm.slurm $mfile pdf 1.0
done
    
