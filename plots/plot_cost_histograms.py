import torch, numpy
import pdb
import matplotlib.pyplot as plt
plt.ion()

eval_dir = '/home/mbhenaff/scratch/models/eval/'

mfiles, loss = [], []
mfiles += 'model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1-nz=32-beta=0.0-nmix=1-warmstart=1.model-nbatches=200-npred=200-nsample=100-sampling=knn-topz=100.eval'
mfiles += 'model=fwd-cnn-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=mult-nz=32-beta=0.0-warmstart=0.model-nbatches=200-npred=200-nsample=1.eval'
mfiles += 'model=fwd-cnn-vae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=128-decoder=0-combine=add-gclip=1.0-nz=32-beta=0.0001-warmstart=1.model-nbatches=200-npred=10-nsample=100.eval'

for mfile in mfiles:
    loss.append(torch.load(eval_dir + mfile + '/loss.pth'))

n_sample = 100
T = 200
#def compute_cost_histograms(loss, n_sample=100, T=200):
true_costs = loss['true_costs']
pred_costs = loss['pred_costs']
true_costs = true_costs.view(-1, 1, T, 2)
pred_costs = pred_costs.view(-1, n_sample, T, 2)
true_costs = true_costs[:, :, :, 0].mean(2).squeeze().numpy()
pred_costs = pred_costs[:, :, :, 0].mean(2).numpy()
#    true_costs = true_costs[:, :, :, 0].max(2)[0].squeeze().numpy()
#    pred_costs = pred_costs[:, :, :, 0].max(2)[0].numpy()

cutoff = 0.2
indx = []
k = 10
bins = numpy.linspace(0, 1, num=k+2)
bins2=numpy.linspace(0, 1, num=50)
for i in range(k):
    indx = (bins[i] < true_costs) & (true_costs <= bins[i+1])
    plt.figure(1)
    a=plt.hist(pred_costs[indx, :].flatten(), bins2, alpha=0.6, normed=True)
    plt.figure(2)
    plt.plot(a[0])
    

'''
bins=numpy.linspace(0, 1, num=100)
a2=plt.hist(pred_costs[indx2, :].flatten(), bins, alpha=0.6, normed=True)
a3=plt.hist(pred_costs[indx3, :].flatten(), bins, alpha=0.6, normed=True)
a4=plt.hist(pred_costs[indx4, :].flatten(), bins, alpha=0.6, normed=True)
a5=plt.hist(pred_costs[indx5, :].flatten(), bins, alpha=0.6, normed=True)
plt.figure()
plt.plot(a2[0])
plt.plot(a3[0])
plt.plot(a4[0])
plt.plot(a5[0])
#    plt.hist(pred_costs[indx_high==1, :].flatten(), bins, alpha=0.8, normed=True)
#    plt.hist(pred_costs[indx_low==1, :].flatten(), bins, alpha=0.8, normed=True)
plt.legend(['i1', 'i2', 'i3', 'i4', 'i5']) 
plt.title('Distribution of predicted costs for different simulations')
#plt.show()

#compute_cost_histograms(loss1, n_sample=100)
    
'''
