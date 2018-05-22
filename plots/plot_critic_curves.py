import matplotlib.pyplot as plt
import torch
import numpy, scipy, pdb
import scipy.stats



path = '/home/mbhenaff/scratch/models/eval_critics2/'

vae1 = 'model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=1e-05-warmstart=1.model'
vae2 = 'model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=0.0001-warmstart=1.model'
cnn = 'model=fwd-cnn-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-nz=32-beta=0.0-warmstart=0.model'
ae = 'model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1-nz=32-beta=0.0-nmix=1-warmstart=1.model'


def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    time_steps = [i+1 for i in range(len(mean))]
    plt.fill_between(time_steps, ub, lb,
                     color=color_shading, alpha=0.2)
    # plot the mean on top
    plt.plot(time_steps, mean, color_mean)




def mean_confidence_interval(data, confidence=0.95):
    n = data.shape[0]
    m, se = numpy.mean(data, 0), scipy.stats.sem(data, 0)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


nsteps=100

'''
# load VAE
f = []
for seed in [1, 2, 3, 4]:
    f.append(torch.load(path + vae1 + ('/critic-nfeature=128-nhidden=128-lrt=0.0001-sampling=fp-seed={}.model.curves'.format(seed))).get('valid_loss'))
f=numpy.stack(f)[:, :nsteps]
mean, lo, hi = mean_confidence_interval(f)
plot_mean_and_CI(mean, lo, hi, color_mean='r', color_shading='r')
'''

# load VAE
f = []
for seed in [1, 2, 3, 4]:
    f.append(torch.load(path + vae1 + ('/critic-nfeature=128-nhidden=128-lrt=0.0001-sampling=fp-seed={}.model.curves'.format(seed))).get('valid_loss'))
f=numpy.stack(f)[:, :nsteps]
pdb.set_trace()
mean, lo, hi = mean_confidence_interval(f)
plot_mean_and_CI(mean, lo, hi, color_mean='r--', color_shading='r')


# load CNN
f = []
for seed in [1, 2, 3, 4]:
    f.append(torch.load(path + cnn + ('/critic-nfeature=128-nhidden=128-lrt=0.0001-sampling=fp-seed={}.model.curves'.format(seed))).get('valid_loss'))
f=numpy.stack(f)[:, :nsteps]
mean, lo, hi = mean_confidence_interval(f)
plot_mean_and_CI(mean, lo, hi, color_mean='g', color_shading='g')


# load AE
f = []
for seed in [1, 2, 3, 4]:
    f.append(torch.load(path + ae + ('/critic-nfeature=128-nhidden=128-lrt=0.0001-sampling=fp-seed={}.model.curves'.format(seed))).get('valid_loss'))
f=numpy.stack(f)[:, :nsteps]
mean, lo, hi = mean_confidence_interval(f)
plot_mean_and_CI(mean, lo, hi, color_mean='magenta', color_shading='magenta')

# load AE
f = []
for seed in [1, 2, 3, 4]:
    f.append(torch.load(path + ae + ('/critic-nfeature=128-nhidden=128-lrt=0.0001-sampling=knn-seed={}.model-density=0.001.curves'.format(seed))).get('valid_loss'))
f=numpy.stack(f)[:, :nsteps]
mean, lo, hi = mean_confidence_interval(f)
plot_mean_and_CI(mean, lo, hi, color_mean='purple', color_shading='purple')




