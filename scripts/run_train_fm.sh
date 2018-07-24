#!/bin/bash

rm *.err
rm *.out

for dataset in i80; do
    for model in fwd-cnn-ten3; do 
        for lrt in 0.0001; do 
            for nfeature in 256; do 
                for warmstart in 1; do 
                    for ncond in 20; do 
                        for npred in 20; do 
                            for nz in 32; do 
                                for beta in 0.0; do 
                                    for dropout in 0.5; do 
                                        for zeroact in 0; do 
                                            for layers in 3; do 
                                                for nhidden in 128; do 
                                                    for a_noise in 0.0; do 
                                                        for bsize in 64; do
                                                            for seed in 2 3 4 5 6 7 8 9; do 
                                                                sbatch submit_train_fm.slurm $model $lrt $nfeature $warmstart $ncond $npred $beta $nz $dataset $dropout $zeroact $layers $nhidden $a_noise $bsize $seed
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
                done
            done
        done
    done
done
