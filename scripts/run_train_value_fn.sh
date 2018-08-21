for npred in 20 50; do 
    for dropout in 0.0; do 
        for gamma in 0.99 0.97; do 
            for nsync in 5 10; do 
                sbatch submit_train_value_fn.slurm $npred $dropout $gamma $nsync
            done
        done
    done
done
