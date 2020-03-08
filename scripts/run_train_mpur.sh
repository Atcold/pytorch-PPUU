# Allows named arguments
set -k

for npred in 30; do
    for batch_size in 6; do
        for u_reg in 0.05; do
            for lambda_a in 0.0; do
                for z_updates in 0; do
                    for lrt_z in 0; do
                        for lambda_l in 0.2; do
                            for lambda_o in 1.0; do
                                for infer_z in 0; do
                                    for seed in 1 2 3; do
                                        sbatch submit_train_mpur.slurm \
                                            npred=$npred \
                                            u_reg=$u_reg \
                                            lrt_z=$lrt_z \
                                            z_updates=$z_updates \
                                            batch_size=$batch_size \
                                            lambda_l=$lambda_l \
                                            lambda_o=$lambda_o \
                                            infer_z=$infer_z \
                                            lambda_a=$lambda_a \
                                            seed=$seed
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
