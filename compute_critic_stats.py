import torch
import models2 as models
import scipy.stats


path = '/home/mbhenaff/scratch/models/eval_critics2/joint/'

scores = []
for seed in [i+1 for i in range(20)]:
    fname = f'{path}/critic-nfeature=128-nhidden=128-lrt=0.0001-sampling=pdf-seed={seed}.model.curves'
    print(f'[loading {fname}]')
    results = torch.load(fname)
    scores.append(torch.stack(results['valid_scores'][:20]))

s1 = scores
S = 1 - torch.stack(s1)
ae = S[:, :, 0]
vae = S[:, :, 2]
cnn = S[:, :, 1]
timesteps = [5, 10, 15, 20]
for t in timesteps:
    ae_t = ae[:, t-1]
    vae_t = vae[:, t-1]
    cnn_t = cnn[:, t-1]
#    p = scipy.stats.ttest_ind(ae_t.numpy(), vae_t.numpy())
    p = scipy.stats.ranksums(ae_t.numpy(), vae_t.numpy())
    print(f't={t} | CNN: {cnn_t.mean():0.5f} ({cnn_t.std():0.5f}), VAE: {vae_t.mean():0.5f} ({vae_t.std():0.5f}), AE: {ae_t.mean():0.5f} ({ae_t.std():0.5f}), p={p}')
    
