#!/bin/bash

for seed in {11..20}; do 
    sbatch submit_generate_data.slurm $seed
done
