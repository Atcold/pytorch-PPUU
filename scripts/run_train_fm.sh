#!/bin/bash

rm *.err
rm *.out

for model in fwd-cnn-een-fp; do 
    for lrt in 0.0001; do 
        for nfeature in 128; do 
            for nhidden in 100; do 
                for ncond in 10; do 
                    for npred in 20; do 
                        for nz in 1 2 4 8 16; do 
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
