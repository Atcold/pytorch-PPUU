# Allows named arguments
set -k

#for policy in ${target_lane_policies[*]}; do
#policy_dir="/misc/vlgscratch4/LecunGroup/nvidia-collab/yairschiff/pytorch-PPUU/models_learned_cost/policy_networks"
#for tl in 0.9 0.8 0.7; do
#    for policy in $policy_dir/*lambdatl=${tl}*.model; do
#        echo "Working on policy: ${policy:103}..."
#        sbatch \
#        --output ../logs/target_lane_learned_cost/planning_results/${policy:103}.out \
#         --error ../logs/target_lane_learned_cost/planning_results/${policy:103}.err \
#        submit_eval_mpur_tl.slurm \
#            model_dir=models_learned_cost \
#            policy=${policy:103}
#    done
#done

# Baseline
policy_dir="/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v13/policy_networks"
for policy in $policy_dir/MPUR-policy-deterministic-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=0-learnedcost=False-seed=*-novaluestep*.model; do
      echo "Working on policy: ${policy:70}..."
      sbatch \
      --output ../logs/target_lane_learned_cost/baseline/planning_results/${policy:70}.out \
      --error ../logs/target_lane_learned_cost/baseline/planning_results/${policy:70}.err \
      submit_eval_mpur_tl.slurm \
      model_dir="/misc/vlgscratch4/LecunGroup/nvidia-collab/models_v13/" \
      policy=${policy:70}
done
