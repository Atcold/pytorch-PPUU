#!/bin/bash

rm datagen*

for seed in 11 13 25 40; do 
    sbatch submit_generate_data.slurm $seed
done
