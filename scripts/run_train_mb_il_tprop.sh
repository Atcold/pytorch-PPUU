#!/bin/bash

for targetprop in 1; do 
    for size in 256; do 
        for npred in 20 50; do 
            for lambda_c in 0.0 0.2; do 
                for gamma in 0.98; do 
                    for seed in 1; do 
                        sbatch submit_train_mb_il.slurm $size $size $seed $npred $lambda_c $targetprop $gamma
                    done
                done
            done
        done
    done
done

