#!/bin/bash
# Allows named arguments
set -k

rm *.err
rm *.out

for model in fwd-cnn-vae-fp; do 
    for lrt in 0.0001; do 
        for nfeature in 256; do 
            for warmstart in 1; do 
                for ncond in 20; do 
                    for npred in 20; do 
                        for nz in 32; do 
                            for beta in 0.000001; do 
                                for z_dropout in 0.5; do 
                                    for layers in 3; do 
                                        for bsize in 64; do
                                            for seed in 1; do 
                                                for dropout in 0.1; do
                                                    sbatch submit_train_fm.slurm \
                                                    model=$model \
                                                    lrt=$lrt \
                                                    nfeature=$nfeature \
                                                    warmstart=$warmstart \
                                                    ncond=$ncond \
                                                    npred=$npred \
                                                    beta=$beta \
                                                    nz=$nz \
                                                    z_dropout=$z_dropout\
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
