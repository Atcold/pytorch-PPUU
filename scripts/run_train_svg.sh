#!/bin/bash

for npred in 40; do 
    for bsize in 6; do 
        for lambda_u in 1.0 0.5; do 
            for lambda_a in 0.1 0.05 0.01; do 
                for z_updates in 0; do 
                    for lrt_z in 0.0; do 
                        sbatch submit_train_svg.slurm $npred $lambda_u $lrt_z $z_updates $bsize $lambda_a
                    done
                done
            done
        done
    done
done
