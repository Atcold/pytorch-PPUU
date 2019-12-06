#!/bin/bash
# Allows named arguments
set -k

for model in fwd-cnn; do
    for lrt in 0.0001; do
        for nfeature in 256; do
            for warmstart in 0; do
                for ncond in 20; do
                    for npred in 20; do
                        for nz in 0; do
                            for beta in 0; do
                                for z_dropout in 0; do
                                    for layers in 3; do
                                        for bsize in 8; do
                                            for seed in 1; do
                                                for dropout in 0.1; do
                                                    for l2reg in 0.0 0.01 0.001 0.0001; do
                                                        sbatch submit_train_fm.slurm \
                                                        model=$model \
                                                        lrt=$lrt \
                                                        nfeature=$nfeature \
                                                        warmstart=$warmstart \
                                                        ncond=$ncond \
                                                        npred=$npred \
                                                        beta=$beta \
                                                        nz=$nz \
                                                        z_dropout=$z_dropout \
                                                        layers=$layers \
                                                        batch_size=$bsize \
                                                        seed=$seed \
                                                        dropout=$dropout
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
