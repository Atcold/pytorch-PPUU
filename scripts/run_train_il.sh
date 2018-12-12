#!/bin/bash

rm *.err
rm *.out

for model in policy-il-mdn; do 
    for lrt in 0.0005 0.0001; do 
        for nhidden in 256; do 
            for ncond in 20; do 
                for npred in 1; do 
                    for nz in 0; do 
                        for beta in 1; do 
                            for nmix in 1; do 
                                for seed in 1 2 3; do 
                                    sbatch submit_train_il.slurm $model $lrt $nhidden $ncond $npred $beta $nz $nmix $seed
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
