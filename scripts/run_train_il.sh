#!/bin/bash

rm *.err
rm *.out

for model in policy-cnn-een; do 
    for lrt in 0.0001; do 
        for nhidden in 100; do 
            for ncond in 10; do 
                for npred in 10 20; do 
                    for nz in 1 2 4 8 16; do 
                        for beta in 1; do 
                            for nshards in 20; do 
                                sbatch submit_train_il.slurm $model $lrt $nhidden $ncond $npred $beta $nshards $nz
                            done
                        done
                    done
                done
            done
        done
    done
done
