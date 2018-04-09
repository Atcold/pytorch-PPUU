#!/bin/bash

rm *.err
rm *.out

for dataset in i80; do
    for model in fwd-cnn-vae-fp; do 
        for lrt in 0.0001; do 
            for nfeature in 96; do 
                for warmstart in 1; do 
                    for ncond in 10; do 
                        for npred in 20; do 
                            for nz in 32; do 
                                for beta in 0.1 0.01 0.001 0.0005 0.0003 0.0002 0.0001; do 
                                    sbatch submit_train_fm.slurm $model $lrt $nfeature $warmstart $ncond $npred $beta $nz $dataset
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
