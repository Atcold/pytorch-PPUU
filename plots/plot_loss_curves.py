import torch
import pdb
import matplotlib.pyplot as plt


eval_dir = '/misc/vlgscratch4/LecunGroup/nvidia-collab/models/eval/'


mfiles = []
mfiles += ['model=fwd-cnn-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-warmstart=0.model-npred=200-nsample=1-pdf-usphere=0.eval']
mfiles += ['model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-warmstart=1.model-npred=200-nsample=200-fp-usphere=0.eval']
mfiles += ['model=fwd-cnn-vae-lp-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-beta=0.0001-warmstart=1.model-npred=200-nsample=200-fp-usphere=0.eval']
mfiles += ['model=fwd-cnn-vae-lp-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-beta=0.0002-warmstart=1.model-npred=200-nsample=200-fp-usphere=0.eval']
mfiles += ['model=fwd-cnn-vae-lp-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-beta=0.0003-warmstart=1.model-npred=200-nsample=200-fp-usphere=0.eval']
mfiles += ['model=fwd-cnn-vae-lp-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-beta=0.0005-warmstart=1.model-npred=200-nsample=200-fp-usphere=0.eval']
mfiles += ['model=fwd-cnn-vae-lp-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-beta=0.001-warmstart=1.model-npred=200-nsample=200-fp-usphere=0.eval']

loss = []
for mfile in mfiles:
    x=torch.load(eval_dir + mfile + '/loss.pth')
    y = []
    for s in range(1, 200):
        y.append(torch.mean(torch.min(x[:, :s], dim=1)[0]))
    loss.append(y)
