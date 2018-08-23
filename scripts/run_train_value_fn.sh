for size in 32; do 
    for npred in 50; do 
        for dropout in 0.05 0.1; do 
            for gamma in 0.97 0.99; do 
                for nsync in 1; do 
                    sbatch submit_train_value_fn.slurm $npred $dropout $gamma $nsync $size
                done
            done
        done
    done
done
