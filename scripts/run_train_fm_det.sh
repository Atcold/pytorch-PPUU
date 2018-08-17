#!/bin/bash

for model in fwd-cnn3; do
    for lrt in 0.0001; do
        for nfeature in 256; do
            for warmstart in 0; do
                for ncond in 20; do
                    for npred in 20; do
                        for nz in 0; do
                            for beta in 0; do
                                for z_dropout in 0; do
                                    for zeroact in 0; do
                                        for layers in 3; do
                                            for nhidden in 128; do
                                                for bsize in 64; do
                                                    for seed in 1; do
                                                        for dropout in 0.0 0.05 0.1; do
                                                            for zmult in 0; do
                                                                sbatch submit_train_fm.slurm $model $lrt $nfeature $warmstart $ncond $npred $beta $nz $z_dropout $zeroact $layers $nhidden $bsize $seed $dropout $zmult
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
