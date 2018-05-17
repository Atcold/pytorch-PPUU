import torch
import matplotlib.pyplot as plt
import cv2, pdb, numpy, os
from scipy import misc

dirname = '/home/mikael/work/pytorch-Traffic-Simulator/scratch/model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-warmstart=1.model-npred=200-nsample=20-pdf-usphere=0.eval/videos/'

dirname_baseline = '/home/mikael/work/pytorch-Traffic-Simulator/scratch/model=fwd-cnn-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-warmstart=0.model-npred=200-nsample=10-pdf-usphere=0.eval/videos/'


x = 17
z = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
#z = [0 , 1, 2, 3, 4, 5, 6, 7]
#z = [0, 2, 4, 11, 17, 18]

im_z, diff = [], []
for z_k in z:
    path = dirname + f'/x{x}/z{z_k}/im00000.png'
    print(path)
    im_z.append(numpy.flipud(misc.imread(path).astype('float')))

#im_x = cv2.imread(dirname_baseline + f'/x{x}/z0/im00000.png')
#im_x = cv2.imread(dirname_baseline + f'/x{x}/z0/im00000.png')
im_y = numpy.flipud(misc.imread(dirname_baseline + f'/x{x}/x/im00009.png'))


diff = []
#plot_indx = [0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
plot_indx = [1, 19]
plt.subplot(1, len(plot_indx)+1, 1)

d0 = im_y
d0 += -d0.min()
d0 = d0/d0.max()
plt.subplot(1, len(plot_indx)+1, 1)
plt.imshow(d0)
plt.xticks([])
plt.yticks([])

#d = im_z[1] - im_z[15]
d = -(im_z[1] - im_z[15])
d += -d.min()
d = d/d.max()
plt.subplot(1, len(plot_indx)+1, 2)
plt.xticks([])
plt.yticks([])
plt.imshow(d)


d = -(im_z[19] - im_z[18])
d += -d.min()
d = d/d.max()
plt.subplot(1, len(plot_indx)+1, 3)
plt.xticks([])
plt.yticks([])
plt.imshow(d)




'''
for i in range(len(plot_indx)):
    k = plot_indx[i]
    d = im_z[k] - im_z[k-1]
    d += -d.min()
    d = d/d.max()
    plt.subplot(1, len(plot_indx)+1, i+2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(d)
'''

plt.savefig('gradient_z.eps')
plt.show()
