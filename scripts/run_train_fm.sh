#!/bin/bash

# Pick the map you'd like to generate data for (comment the others)
#MAP="ai"
#MAP="i80"
MAP="us101"
#MAP="lanker"
#MAP="peach"

#rm *.err
#rm *.out

for model in fwd-cnn-vae-fp; do 
    for lrt in 0.0001; do 
        for nfeature in 256; do 
            for warmstart in 0; do # changed from 1 to 0 
                for ncond in 20; do 
                    for npred in 20; do 
                        for nz in 32; do 
                            for beta in 0.000001; do 
                                for z_dropout in 0.5; do 
                                    for layers in 3; do 
                                        for bsize in 8; do
                                            for seed in 1; do 
                                                for dropout in 0.1; do
							sbatch \
								--output ../logs/train_fm_${model}_${MAP}.out \
								--error ../logs/train_fm_${model}_${MAP}.err \
								submit_train_fm.slurm $model $lrt $nfeature $warmstart $ncond $npred $beta $nz $z_dropout $layers $bsize $seed $dropout ${MAP} traffic-data/state-action-cost/data_${MAP}_v0/ models_us101_v2
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
