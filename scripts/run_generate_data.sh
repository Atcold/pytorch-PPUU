#!/bin/bash

for seed in {21..40}; do 
    sbatch submit_generate_data.slurm $seed
done
