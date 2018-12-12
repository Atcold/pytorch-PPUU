#!/bin/bash

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
                                        for bsize in 8; do
                                            for seed in 1; do 
                                                for dropout in 0.1; do 
                                                    for l2reg in 0.001; do 
                                                        sbatch submit_train_fm.slurm $model $lrt $nfeature $warmstart $ncond $npred $beta $nz $z_dropout $layers $bsize $seed $dropout $l2reg
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
