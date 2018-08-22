import matplotlib.pyplot as plt
import numpy, scipy, argparse, pdb
import scipy.stats
import torch


parser = argparse.ArgumentParser()
parser.add_argument('-metric', type=str, default='l2')
parser.add_argument('-path', type=str, default='/home/mbhenaff/projects/pytorch-Traffic-Simulator/scratch/models_v8/eval/')
opt = parser.parse_args()


def best_of_k(x):
    bsize = x.shape[0]
    best_samples = []
    for b in range(bsize):
        mean = numpy.mean(x[b], 1)
        best = numpy.argsort(mean)
        best_samples.append(x[b][best[-1]])
    best_samples = numpy.stack(best_samples)
    return best_samples




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

ssim = []


plot = 0
loss_type = 'loss_i'
npred = 10
nsample = 200

if plot == 0:

    x = torch.load(f'{opt.path}/model=fwd-cnn-vae3-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-nhidden=128-fgeom=1-zeroact=0-zmult=0-dropout=0.0-nz=32-beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model-nbatches=25-npred=200-nsample=200.eval/loss.pth')
    x = x[loss_type].view(100*4, nsample, -1)[:, :, :npred].numpy()
    x=-10*numpy.log(x) / numpy.log(10)
    best = best_of_k(x)
    mean, hi, low = mean_confidence_interval(best)
    plot_mean_and_CI(mean, low, hi, color_mean='b-', color_shading='b')

    x = torch.load(f'{opt.path}/model=fwd-cnn-vae3-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-nhidden=128-fgeom=1-zeroact=0-zmult=0-dropout=0.0-nz=32-beta=1e-05-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model-nbatches=25-npred=200-nsample=200.eval/loss.pth')
    x = x[loss_type].view(100*4, nsample, -1)[:, :, :npred].numpy()
    x=-10*numpy.log(x) / numpy.log(10)
    best = best_of_k(x)
    mean, hi, low = mean_confidence_interval(best)
    plot_mean_and_CI(mean, low, hi, color_mean='magenta', color_shading='magenta')


    x = torch.load(f'{opt.path}/model=fwd-cnn-ten3-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-nhidden=128-fgeom=1-zeroact=0-zmult=0-dropout=0.0-nz=32-beta=0.0-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model-nbatches=25-npred=200-nsample=200-sampling=fp.eval/loss.pth')
    x = x[loss_type].view(100*4, nsample, -1)[:, :, :npred].numpy()
    x=-10*numpy.log(x) / numpy.log(10)
    best = best_of_k(x)
    mean, hi, low = mean_confidence_interval(best)
    plot_mean_and_CI(mean, low, hi, color_mean='red', color_shading='red')


    '''
    x = torch.load(f'{opt.path}/model=fwd-cnn-vae3-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-nhidden=128-fgeom=1-zeroact=0-zmult=0-dropout=0.0-nz=32-beta=0.0001-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model-nbatches=25-npred=200-nsample=200.eval/loss.pth')
    x = x[loss_type].view(100*4, nsample, -1)[:, :, :npred].numpy()
    x=-10*numpy.log(x) / numpy.log(10)
    best = best_of_k(x)
    mean, hi, low = mean_confidence_interval(best)
    plot_mean_and_CI(mean, low, hi, color_mean='cyan', color_shading='cyan')
    '''







title = 'comparison_vae.pdf'
plt.ylabel('Average PSNR', fontsize=18)
plt.xlabel('Time Step', fontsize=18)
plt.xticks([i+1 for i in range(npred)], fontsize=12)
plt.legend(['VAE, beta=1e-06', 'VAE, beta=1e-05', 'TEN'], fontsize=16)
plt.savefig(title)

