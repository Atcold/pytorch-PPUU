#!/bin/bash

MAP="highD"

for i in {01..60}; do  # recording
    sbatch \
        --output ../logs/highD/${MAP}_ts${i}.out \
        --error ../logs/highD/${MAP}_ts${i}.err \
        submit_generate_data_${MAP}.slurm $i
done
