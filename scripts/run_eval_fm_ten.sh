#!/bin/bash

for mfile in /home/mbhenaff/scratch/models_v7/*ten3*-dropout=0.05*beta=0.0*seed=1.model; do
    sbatch submit_eval_fm.slurm $(basename $mfile) fp 1.0
done
    
