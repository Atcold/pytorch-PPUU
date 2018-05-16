import matplotlib.pyplot as plt
import numpy, scipy, argparse, pdb
import scipy.stats
import torch


parser = argparse.ArgumentParser()
parser.add_argument('-metric', type=str, default='l2')
parser.add_argument('-path', type=str, default='/home/mbhenaff/scratch/models/eval/')
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

    x = torch.load(f'{opt.path}/model=fwd-cnn-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-nz=32-beta=0.0-warmstart=0.model-nbatches=100-npred=50-nsample=1.eval/loss.pth')
    x = x[loss_type].view(100*4, -1)[:, :npred].numpy()
    x=-10*numpy.log(x) / numpy.log(10)
    best = x
    mean, hi, low = mean_confidence_interval(best)
    plot_mean_and_CI(mean, low, hi, color_mean='gray', color_shading='gray')


    x = torch.load(f'{opt.path}/model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1-nz=32-beta=0.0-nmix=1-warmstart=1.model-nbatches=100-npred=50-nsample=200-sampling=fp.eval/loss.pth')
    x = x[loss_type].view(100*4, nsample, -1)[:, :, :npred].numpy()
    x=-10*numpy.log(x) / numpy.log(10)
    best = best_of_k(x)
    mean, hi, low = mean_confidence_interval(best)
    plot_mean_and_CI(mean, low, hi, color_mean='magenta', color_shading='magenta')


    '''
    x = torch.load(f'{opt.path}/model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1-nz=32-beta=0.0-nmix=1-warmstart=1.model-nbatches=100-npred=50-nsample=200-sampling=knn-density=0.0001.eval/loss.pth')
    x = x['loss_i'].view(100*4, nsample, -1)[:, :, :npred].numpy()
    x=-10*numpy.log(x) / numpy.log(10)
    best = best_of_k(x)
    mean, hi, low = mean_confidence_interval(best)
    plot_mean_and_CI(mean, low, hi, color_mean='b--', color_shading='b')


    x = torch.load(f'{opt.path}/model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1-nz=32-beta=0.0-nmix=1-warmstart=1.model-nbatches=100-npred=50-nsample=200-sampling=knn-density=0.0005.eval/loss.pth')
    x = x['loss_i'].view(100*4, nsample, -1)[:, :, :npred].numpy()
    x=-10*numpy.log(x) / numpy.log(10)
    best = best_of_k(x)
    mean, hi, low = mean_confidence_interval(best)
    plot_mean_and_CI(mean, low, hi, color_mean='r', color_shading='r')


    x = torch.load(f'{opt.path}/model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1-nz=32-beta=0.0-nmix=1-warmstart=1.model-nbatches=100-npred=50-nsample=200-sampling=knn-density=0.001.eval/loss.pth')
    x = x[loss_type].view(100*4, nsample, -1)[:, :, :npred].numpy()
    x=-10*numpy.log(x) / numpy.log(10)
    best = best_of_k(x)
    mean, hi, low = mean_confidence_interval(best)
    plot_mean_and_CI(mean, low, hi, color_mean='g', color_shading='g')

    x = torch.load(f'{opt.path}/model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1-nz=32-beta=0.0-nmix=1-warmstart=1.model-nbatches=100-npred=50-nsample=200-sampling=knn-density=0.002.eval/loss.pth')
    x = x[loss_type].view(100*4, nsample, -1)[:, :, :npred].numpy()
    x=-10*numpy.log(x) / numpy.log(10)
    best = best_of_k(x)
    mean, hi, low = mean_confidence_interval(best)
    plot_mean_and_CI(mean, low, hi, color_mean='g--', color_shading='g')

    '''
    x = torch.load(f'{opt.path}/model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1-nz=32-beta=0.0-nmix=1-warmstart=1.model-nbatches=100-npred=50-nsample=200-sampling=knn-density=0.005.eval/loss.pth')
    x = x[loss_type].view(100*4, nsample, -1)[:, :, :npred].numpy()
    x=-10*numpy.log(x) / numpy.log(10)
    best = best_of_k(x)
    mean, hi, low = mean_confidence_interval(best)
    plot_mean_and_CI(mean, low, hi, color_mean='blue', color_shading='blue')


    x = torch.load(f'{opt.path}/model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=0.0001-warmstart=1.model-nbatches=100-npred=10-nsample=200.eval/loss.pth')
    x = x[loss_type].view(100*4, nsample, -1)[:, :, :npred].numpy()
    x=-10*numpy.log(x) / numpy.log(10)
    best = best_of_k(x)
    mean, hi, low = mean_confidence_interval(best)
    plot_mean_and_CI(mean, low, hi, color_mean='r', color_shading='r')
    pdb.set_trace()




    title = 'ae_comparison.pdf'




plt.ylabel('Average PSNR', fontsize=16)
plt.xlabel('Time Step', fontsize=16)

#plt.savefig(title)

