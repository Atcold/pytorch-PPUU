#!/bin/bash

for seed in {11..30}; do 
    sbatch submit_generate_data.slurm $seed
done
