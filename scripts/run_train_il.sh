#!/bin/bash

rm *.err
rm *.out

for model in policy-cnn; do 
    for lrt in 0.001 0.0001; do 
        for nhidden in 100; do 
            for ncond in 4; do 
                for npred in 20 50; do 
                    for beta in 0.001; do 
                        for nshards in 20; do 
                            sbatch submit_train_il.slurm $model $lrt $nhidden $ncond $npred $beta $nshards
                        done
                    done
                done
            done
        done
    done
done
