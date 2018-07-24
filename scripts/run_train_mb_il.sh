#!/bin/bash

for targetprop in 0; do 
    for size in 256; do 
        for npred in 50 100 200 400; do 
            for lambda_c in 0.0; do 
                for gamma in 0.997; do 
                    for seed in 1; do 
                        for bsize in 16; do 
                            for curriculum_length in 10; do 
                                for subsample in -1; do 
                                    for lambda_h in 0.0; do
                                        sbatch submit_train_mb_il.slurm $size $size $seed $npred $lambda_c $targetprop $gamma $bsize $curriculum_length $subsample $lambda_h
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

