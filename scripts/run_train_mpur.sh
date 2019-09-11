# Allows named arguments
set -k

for npred in 30; do
    for batch_size in 6; do
        for u_reg in 0.05; do
            for lambda_a in 0.0; do
                for lambda_l in 0.2; do
                    for lambda_tl in 0.1 0.2 0.3 0.4 0.5; do
                        for seed in 1 2 3; do
                            sbatch \
                              --output ../logs/target_lane/train_MPUR_seed${seed}_lambdatl${lambda_tl}.out \
                              --error ../logs/target_lane/train_MPUR_seed${seed}_lambdatl${lambda_tl}.err \
                              submit_train_mpur.slurm \
                              npred=$npred \
                              u_reg=$u_reg \
                              lrt_z=$lrt_z \
                              z_updates=$z_updates \
                              batch_size=$batch_size \
                              lambda_l=$lambda_l \
                              infer_z=$infer_z \
                              lambda_a=$lambda_a \
                              lambda_tl=$lambda_tl \
                              seed=$seed
                        done
                    done
                done
            done
        done
    done
done
