# Allows named arguments
set -k


# Target lane policy
policy_dir="/misc/vlgscratch4/LecunGroup/nvidia-collab/yairschiff/pytorch-PPUU/models_learned_cost/policy_networks"
save_dir="planning_results_look"
for tl in 1.0; do
  for ll in 0.0; do
    for policy in $policy_dir/*lambdal=${ll}*lambdatl=${tl}*.model; do
      echo "Working on policy: ${policy:103}..."
      sbatch \
        --output ../logs/target_lane_learned_cost/planning_results_look/${policy:103}.out \
        --error ../logs/target_lane_learned_cost/planning_results_look/${policy:103}.err \
        submit_eval_mpur_tl.slurm \
          model_dir="models_learned_cost" \
          policy=${policy:103} \
          save_dir=${save_dir}
    done
  done
done

# Baseline
#policy_dir="/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v13/policy_networks"
#for policy in $policy_dir/MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=False-seed=*-novaluestep*.model; do
#      echo "Working on policy: ${policy:70}..."
#      sbatch \
#      --output ../logs/target_lane_learned_cost/baseline/planning_results/${policy:70}.out \
#      --error ../logs/target_lane_learned_cost/baseline/planning_results/${policy:70}.err \
#      submit_eval_mpur_tl.slurm \
#      model_dir="/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v13/" \
#      policy=${policy:70}
#done
