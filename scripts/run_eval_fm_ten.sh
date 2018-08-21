#!/bin/bash

for mfile in /home/mbhenaff/scratch/models_v8/*vae3*-dropout=*beta=1e-06*zdropout=*-seed=1.step200000.model; do
    sbatch submit_eval_fm.slurm $(basename $mfile) fp 1.0
done
    
