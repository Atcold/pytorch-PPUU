#!/bin/bash

rm *.err
rm *.out

for dataset in i80; do
    for model in fwd-cnn; do 
        for lrt in 0.0001; do 
            for nfeature in 96; do 
                for nhidden in 100; do 
                    for ncond in 10; do 
                        for npred in 30 40 50; do 
                            for nz in 1; do 
                                for beta in 1.0; do 
                                    sbatch submit_train_fm.slurm $model $lrt $nfeature $nhidden $ncond $npred $beta $nz $dataset
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
