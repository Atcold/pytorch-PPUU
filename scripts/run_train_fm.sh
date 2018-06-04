#!/bin/bash

rm *.err
rm *.out

for dataset in i80; do
    for model in fwd-cnn-ten; do 
        for lrt in 0.0001; do 
            for nfeature in 128; do 
                for warmstart in 0 1; do 
                    for ncond in 10; do 
                        for npred in 20; do 
                            for nz in 16 32; do 
                                for beta in 0.0; do 
                                    for dropout in 0.0 0.25 0.5; do 
                                        sbatch submit_train_fm.slurm $model $lrt $nfeature $warmstart $ncond $npred $beta $nz $dataset $dropout
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
