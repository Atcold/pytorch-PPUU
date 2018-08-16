#!/bin/bash

for npred in 20 50; do 
    for n_rollouts in 10 20; do 
        for lrt in 0.5 1.0; do 
            for niter in 10; do 
                sbatch submit_plan.slurm $n_rollouts $npred 0.0 $lrt $niter
            done
        done
    done
done
