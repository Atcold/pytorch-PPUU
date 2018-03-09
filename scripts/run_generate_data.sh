#!/bin/bash

rm datagen*

for seed in {21..30}; do 
    sbatch submit_generate_data.slurm $seed
done
