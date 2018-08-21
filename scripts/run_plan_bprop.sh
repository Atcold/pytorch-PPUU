#!/bin/bash

for method in bprop; do 
    for niter in 5 10; do 
        for lrt in 0.5; do 
            for u_reg in 0.1; do 
                sbatch submit_plan_bprop.slurm $method $niter $lrt $u_reg
            done
        done
    done
done
