#!/bin/bash
# Allows named arguments
set -k
-mfile $1 -batch_size 4 -npred 200 -sampling $2 -n_sample 10 -n_batches 10 -save_video 1
for mfile in /home/mbhenaff/scratch/models_v11/*vae*-dropout=*beta=1e-06*-seed=1.step200000.model; do
    sbatch submit_eval_fm.slurm mfile=$(basename $mfile) sampling='fp'
done
    
