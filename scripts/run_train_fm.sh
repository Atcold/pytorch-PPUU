#!/bin/bash

rm *.err
rm *.out

for model in fwd-cnn2; do 
    for lrt in 0.001 0.0001; do 
        for nfeature in 128; do 
            for nhidden in 100; do 
                for ncond in 4; do 
                    for npred in 50; do 
                        for nz in 8; do 
                            for beta in 0.001; do 
                                sbatch submit_train_fm.slurm $model $lrt $nfeature $nhidden $ncond $npred $beta $nz
                            done
                        done
                    done
                done
            done
        done
    done
done
