#!/bin/bash

sbatch submit_train_critic.slurm model=fwd-cnn-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-warmstart=0.model
sbatch submit_train_critic.slurm model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-warmstart=1.model
sbatch submit_train_critic.slurm model=fwd-cnn-vae-lp-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-beta=0.0001-warmstart=1.model
sbatch submit_train_critic.slurm model=fwd-cnn-vae-lp-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-beta=0.0002-warmstart=1.model
sbatch submit_train_critic.slurm model=fwd-cnn-vae-lp-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-beta=0.0003-warmstart=1.model
sbatch submit_train_critic.slurm model=fwd-cnn-vae-lp-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-beta=0.0005-warmstart=1.model
sbatch submit_train_critic.slurm model=fwd-cnn-vae-lp-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-beta=0.001-warmstart=1.model
sbatch submit_train_critic.slurm model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-beta=0.0001-zsphere=1-zmult=1-gclip=-1-warmstart=1.model
sbatch submit_train_critic.slurm model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-beta=0.0002-zsphere=1-zmult=1-gclip=-1-warmstart=1.model
sbatch submit_train_critic.slurm model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-beta=0.0003-zsphere=1-zmult=1-gclip=-1-warmstart=1.model
sbatch submit_train_critic.slurm model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-beta=0.0005-zsphere=1-zmult=1-gclip=-1-warmstart=1.model
sbatch submit_train_critic.slurm model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-beta=0.001-zsphere=1-zmult=1-gclip=-1-warmstart=1.model
sbatch submit_train_critic.slurm model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-beta=0.01-zsphere=1-zmult=1-gclip=-1-warmstart=1.model
sbatch submit_train_critic.slurm model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-beta=0.1-zsphere=1-zmult=1-gclip=-1-warmstart=1.model
