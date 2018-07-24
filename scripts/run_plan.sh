#!/bin/bash

for npred in 50 100 200; do 
    for lrt in 0.1 0.01; do 
        for opt_z in 0; do 
            for n_rollouts in 20; do 
                for nexec in -1; do 
                    sbatch submit_plan.slurm $npred $lrt $opt_z $n_rollouts $nexec
                done
            done
        done
    done
done
