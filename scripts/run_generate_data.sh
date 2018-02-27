#!/bin/bash

for seed in {1..10}; do 
    sbatch submit_generate_data.slurm $seed
done
