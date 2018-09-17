#!/bin/bash

for npred in 20; do 
    for bsize in 12; do 
        for u_reg in 0.0 0.01 0.05; do 
            for lambda_a in 0.0; do 
                for z_updates in 0; do 
                    for lrt_z in 0.0; do 
                        for lambda_l in 0.0 0.1 0.2; do 
                            sbatch submit_train_svg.slurm $npred $u_reg $lrt_z $z_updates $bsize $lambda_l
                        done
                    done
                done
            done
        done
    done
done
