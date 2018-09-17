#!/bin/bash

for method in bprop; do 
    for niter in 5; do 
        for lrt in 0.1; do 
            for lane_cost in 0.1; do 
                for u_reg in 0.0 0.1; do 
                    for u_hinge in 0.0; do 
                        for buffer in 1; do 
                            sbatch submit_plan_bprop.slurm $method $niter $lrt $u_reg $u_hinge $buffer $lane_cost
                        done
                    done
                done
            done
        done
    done
done
