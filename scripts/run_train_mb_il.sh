#!/bin/bash

for targetprop in 0; do 
    for size in 256; do 
        for npred in 1; do 
            for lambda_c in 0.0; do 
                for gamma in 0.99; do 
                    for seed in 2 3; do 
                        for bsize in 16; do 
                            for curriculum_length in 1; do 
                                for subsample in 1; do 
                                    for lambda_h in 0; do
                                        for context_dim in 1; do 
                                            sbatch submit_train_mb_il.slurm $size $size $seed $npred $lambda_c $targetprop $gamma $bsize $curriculum_length $subsample $lambda_h $context_dim 
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
done

