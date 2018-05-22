#!/bin/bash


sbatch submit_eval_fm.slurm model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=0.01-warmstart=1.model 0 0 
sbatch submit_eval_fm.slurm model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=0.001-warmstart=1.model 0 0
sbatch submit_eval_fm.slurm model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=0.0001-warmstart=1.model 0 0
sbatch submit_eval_fm.slurm model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=1e-05-warmstart=1.model 0 0
#sbatch submit_eval_fm.slurm model=fwd-cnn-vae-lp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=0.01-warmstart=1.model 0 0 
#sbatch submit_eval_fm.slurm model=fwd-cnn-vae-lp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=0.001-warmstart=1.model 0 0
#sbatch submit_eval_fm.slurm model=fwd-cnn-vae-lp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=0.0001-warmstart=1.model 0 0
#sbatch submit_eval_fm.slurm model=fwd-cnn-vae-lp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=1e-05-warmstart=1.model 0 0


sbatch submit_eval_fm.slurm model=fwd-cnn-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-nz=32-beta=0.0-warmstart=0.model 0 0 
#sbatch submit_eval_fm.slurm model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=0.01-warmstart=1.model 0 0 
#sbatch submit_eval_fm.slurm model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=0.001-warmstart=1.model 0 0 
#sbatch submit_eval_fm.slurm model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=0.0001-warmstart=1.model 0 0 
#sbatch submit_eval_fm.slurm model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=1e-05-warmstart=1.model 0 0 
