#!/bin/bash

rm datagen*

for seed in {1..20}; do 
    sbatch submit_generate_data.slurm $seed
done
