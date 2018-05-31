#!/bin/bash

for i in 0 1 2; do 
    sbatch submit_generate_data.slurm $i
done
