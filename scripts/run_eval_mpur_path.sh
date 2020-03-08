# Allows named arguments
set -k

if [ $# -eq 0 ]
then
    echo "Pass the directory where *.model files are stored"
    exit
fi

model_dir=$1

for f in $model_dir/policy_networks/*0.model; do
    policy=$(basename $f);
    sbatch submit_eval_mpur_path.slurm \
        policy=$policy \
        model_dir=$model_dir
done
