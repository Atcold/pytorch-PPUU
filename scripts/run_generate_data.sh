#!/bin/bash

for seed in {1..30}; do 
    sbatch submit_generate_data.slurm $seed
done
