#!/bin/bash
#
#SBATCH --job-name=train_MPUR
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --constraint="gpu_12gb&pascal"
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11"
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yzs208@nyu.edu

source activate PPUU
cd ../

srun python train_MPUR.py \
    -npred $npred \
    -u_reg $u_reg \
    -lrt_z $lrt_z \
    -z_updates $z_updates \
    -batch_size $batch_size \
    -lambda_l $lambda_l \
    -infer_z $infer_z \
    -lambda_a $lambda_a \
    -seed $seed \
    -model $model \
    -dataset $dataset \
    -data_dir $data_dir \
    -model_dir $model_dir \
    -mfile $mfile

