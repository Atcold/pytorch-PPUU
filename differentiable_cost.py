import torch, numpy as np
from dataloader import DataLoader
import pdb, pickle, glob, argparse
import matplotlib.pyplot as plt
from torch.autograd import Variable
plt.ion()


parser = argparse.ArgumentParser()
# data params
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-ncond', type=int, default=20)
parser.add_argument('-npred', type=int, default=20)
parser.add_argument('-debug', type=int, default=0)
opt = parser.parse_args()





def proximity_cost(images, states, car_size=(6.4, 14.3), green_channel=1, unnormalize=False, s_mean=None, s_std=None):
    SCALE = 0.25
    safe_factor = 1.5
    bsize, npred, nchannels, crop_h, crop_w = images.size(0), images.size(1), images.size(2), images.size(3), images.size(4)
    images = images.view(bsize*npred, nchannels, crop_h, crop_w)
    states = states.view(bsize*npred, 4)

    if unnormalize:
        states *= (1e-8 + dataloader.s_std.view(1, 1, 4).expand(states.size())).cuda()
        states += dataloader.s_mean.view(1, 1, 4).expand(states.size()).cuda()

    speed = states[:, 2:].norm(2, 1) * SCALE #pixel/s
    width, length = car_size # feet
    width *= SCALE * (0.3048*24/3.7) # pixels
    length *= SCALE * (0.3048*24/3.7) # pixels

    safe_distance = torch.abs(speed) * safe_factor + (1*24/3.7) * SCALE  # plus one metre (TODO change)
    

    # Compute x/y minimum distance to other vehicles (pixel version)
    # Account for 1 metre overlap (low data accuracy)
    alpha = 1 * SCALE * (24/3.7)  # 1 m overlap collision
    # Create separable proximity mask


    max_x = np.ceil((crop_h - max(length - alpha, 0)) / 2)
    max_y = np.ceil((crop_w - max(width - alpha, 0)) / 2)

    min_x = torch.clamp(max_x - safe_distance, min=0)
    min_y = np.ceil(crop_w / 2 - width)  # assumes other._width / 2 = self._width / 2
    min_y = torch.tensor(min_y)

    x_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_h))) * crop_h / 2
    x_filter = torch.clamp(x_filter, max=max_x)
    x_filter = x_filter.view(1, crop_h).expand(bsize*npred, crop_h)
    x_filter = torch.max(x_filter, min_x.view(bsize*npred, 1))
    max_x = torch.tensor([max_x])
    max_y = torch.tensor([max_y])

    x_filter = (x_filter - min_x.view(bsize*npred, 1)) / (max_x - min_x).view(bsize*npred, 1)
    y_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_w))) * crop_w / 2
    y_filter = torch.clamp(y_filter, min=min_y.item(), max=max_y.item())
    y_filter = (y_filter - min_y) / (max_y - min_y)
    x_filter = x_filter.cuda()
    y_filter = y_filter.cuda()
    proximity_mask = (torch.bmm(x_filter.view(-1, crop_h, 1), y_filter.view(1, crop_w).expand(bsize*npred, 1, crop_w))).view(bsize, npred, crop_h, crop_w)
    images = images.view(bsize, npred, nchannels, crop_h, crop_w)
    costs = torch.max((proximity_mask * images[:, :, green_channel].float()).view(bsize, npred, -1), 2)[0] 
    return costs, proximity_mask














'''
def proximity_cost(images, states, car_size=(6.4, 14.3)):
    SCALE = 0.25
    safe_factor = 1.5

    speed = state[:, 2:].norm(2) * SCALE #pixel/s
    width, length = car_size # feet
    width *= SCALE * (0.3048*24/3.7) # pixels
    length *= SCALE * (0.3048*24/3.7) # pixels

    safe_distance = torch.abs(speed) * safe_factor + (1*24/3.7) * SCALE  # plus one metre (TODO change)
    

    # Compute x/y minimum distance to other vehicles (pixel version)
    # Account for 1 metre overlap (low data accuracy)
    alpha = 1 * SCALE * (24/3.7)  # 1 m overlap collision
    # Create separable proximity mask

    crop_h, crop_w = image.size(0), image.size(1) 

    max_x = np.ceil((crop_h - max(length - alpha, 0)) / 2)
    max_y = np.ceil((crop_w - max(width - alpha, 0)) / 2)
    max_x = torch.tensor(max_x)
    max_y = torch.tensor(max_y)

    min_x = torch.max(max_x - safe_distance, 0)[0]
    min_y = np.ceil(crop_w / 2 - width)  # assumes other._width / 2 = self._width / 2
    min_y = torch.tensor(min_y)

    x_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_h))) * crop_h / 2
    x_filter = torch.clamp(x_filter, max=max_x)
    x_filter = torch.clamp(x_filter, min=min_x, max=max_x)
    x_filter = (x_filter - min_x) / (max_x - min_x)
    
    y_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_w))) * crop_w / 2
    y_filter = torch.clamp(y_filter, min=min_y, max=max_y)
    y_filter = (y_filter - min_y) / (max_y - min_y)
    proximity_mask = x_filter.view(-1, 1) * y_filter.view(1, -1)
    loss = torch.max(proximity_mask * image.float()) / 255.0
    return loss, proximity_mask
'''

true_costs, new_costs = [], []

dataloader = DataLoader(None, opt, 'i80')
inputs, actions, targets = dataloader.get_batch_fm('train', opt.npred)
inputs, actions, targets = dataloader.get_batch_fm('train', opt.npred)
inputs, actions, targets = dataloader.get_batch_fm('train', opt.npred)
images, states, costs = targets
states *= (1e-8 + dataloader.s_std.view(1, 1, 4).expand(states.size())).cuda()
states += dataloader.s_mean.view(1, 1, 4).expand(states.size()).cuda()
costs = costs[:, :, 0]

images = Variable(images)
images.requires_grad=True
pred_costs, masks = proximity_cost(images, states)

plt.plot(costs.view(-1).cpu().numpy())
plt.plot(pred_costs.data.view(-1).cpu().numpy())

'''
for c in glob.glob('/home/mbhenaff/scratch/traffic-data/state-action-cost/data_i80_v0/trajectories-0400-0415/car*.pkl')[:20]:
    print(c)
    with open(c, 'rb') as f:
        x=pickle.load(f)

    images = x['images']
    states = x['states']
    costs = x['pixel_proximity_cost']

    for t in range(images.size(0)):
        cost_t, mask = proximity_cost(images[t][1], states[t][0], car_size=(6.4, 14.3))
        new_costs.append(cost_t)
        true_costs.append(costs[t])

true_costs = torch.stack(true_costs)
new_costs = torch.stack(new_costs)

#plt.imshow(mask)
plt.figure()
plt.plot(true_costs.view(-1).numpy())
plt.plot(new_costs.view(-1).numpy())
plt.show()
'''
'''
#plt.imshow(mask + images[t][0].float()/255 + images[t][1].float()/255 + images[t][2].float()/255)
'''    


