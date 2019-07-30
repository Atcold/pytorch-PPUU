#!/bin/bash

# Allows named arguments
set -k

# Pick the map you'd like to generate data for (comment the others)
#MAP="ai"
#MAP="i80"
MAP="us101"
#MAP="lanker"
#MAP="peach"

data_dir="traffic-data/state-action-cost/data_${MAP}_v0/"
model_dir="models_${MAP}_v2"

for npred in 30; do
    for batch_size in 6; do
        for u_reg in 0.05; do
            for lambda_a in 0.0; do
                for z_updates in 0; do
                    for lrt_z in 0; do
                        for lambda_l in 0.2; do
                            for infer_z in 0; do
                                for seed in 1 2 3 4 5 6; do
                                    for model in fwd-cnn-vae-fp; do
                                        for mfile in model=fwd-cnn-vae-fp-layers=3-bsize=8-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=0-seed=1.step400000.model; do
                                            sbatch \
                                                --output ../logs/train_mpur_${MAP}_${seed}.out \
                                                --error ../logs/train_mpur_${MAP}_${seed}.err \
                                                submit_train_mpur.slurm \
                                                npred=$npred \
                                                u_reg=$u_reg \
                                                lrt_z=$lrt_z \
                                                z_updates=$z_updates \
                                                batch_size=$batch_size \
                                                lambda_l=$lambda_l \
                                                infer_z=$infer_z \
                                                lambda_a=$lambda_a \
                                                seed=$seed \
                                                model=$model \
                                                dataset=$MAP \
                                                data_dir=$data_dir \
                                                model_dir=$model_dir \
                                                mfile=$mfile
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
