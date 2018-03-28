import torch
import os, json, pdb, math, numpy
from datetime import datetime
from sklearn import decomposition
from sklearn.manifold import TSNE

# Logging function
def log(fname, s):
    if not os.path.isdir(os.path.dirname(fname)):
            os.system("mkdir -p " + os.path.dirname(fname))
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()

def grad_norm(net):
    total_norm = 0
    for p in net.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object

def log_pdf(z, mu, sigma):
    a = 0.5*torch.sum(((z-mu)/sigma)**2, 1)
    b = torch.log(2*math.pi*torch.prod(sigma, 1))
    loss = a.squeeze() + b.squeeze()
    return torch.mean(loss)

# embed Z distribution as well as some special z's (ztop) using PCA and tSNE. 
# Useful for visualizing predicted z vectors. 
def embed(Z, ztop):
    bsize = ztop.shape[0]
    nsamples = ztop.shape[1]
    dim = ztop.shape[2]
    ztop = ztop.reshape(bsize*nsamples, dim)
    pca = decomposition.PCA(n_components=2)
    pca.fit(Z)
    Z_pca = pca.transform(Z)
    ztop_pca = pca.transform(ztop).reshape(bsize, nsamples, 2)
    Z_all=numpy.concatenate((ztop, Z), axis=0)
    Z_all_tsne = TSNE(n_components=2).fit_transform(Z_all)
    ztop_tsne = Z_all_tsne[0:bsize*nsamples].reshape(bsize, nsamples, 2)
    Z_tsne = Z_all_tsne[bsize*nsamples:]
    return {'Z_pca': Z_pca, 'ztop_pca': ztop_pca, 'Z_tsne': Z_tsne, 'ztop_tsne': ztop_tsne}
