import torch, numpy as np
from dataloader import DataLoader
import pdb, pickle, glob, argparse
import matplotlib.pyplot as plt
from torch.autograd import Variable
plt.ion()

def proximity_cost(images, states, car_size=[[6.4, 14.3]], green_channel=1, unnormalize=False, s_mean=None, s_std=None, return_argmax=False):
    car_size = torch.tensor(car_size)
    SCALE = 0.25
    safe_factor = 1.5
    bsize, npred, nchannels, crop_h, crop_w = images.size(0), images.size(1), images.size(2), images.size(3), images.size(4)
    images = images.view(bsize*npred, nchannels, crop_h, crop_w)
    states = states.view(bsize*npred, 4)

    if unnormalize:
        states *= (1e-8 + dataloader.s_std.view(1, 1, 4).expand(states.size())).cuda()
        states += dataloader.s_mean.view(1, 1, 4).expand(states.size()).cuda()

    speed = states[:, 2:].norm(2, 1) * SCALE #pixel/s
    width, length = car_size[:, 0], car_size[:, 1] # feet
    width *= SCALE * (0.3048*24/3.7) # pixels
    length *= SCALE * (0.3048*24/3.7) # pixels

    safe_distance = torch.abs(speed) * safe_factor + (1*24/3.7) * SCALE  # plus one metre (TODO change)

    # Compute x/y minimum distance to other vehicles (pixel version)
    # Account for 1 metre overlap (low data accuracy)
    alpha = 1 * SCALE * (24/3.7)  # 1 m overlap collision
    # Create separable proximity mask

    max_x = torch.ceil((crop_h - torch.clamp(length - alpha, min=0)) / 2)
    max_y = torch.ceil((crop_w - torch.clamp(width - alpha, min=0)) / 2)
    max_x = max_x.view(bsize, 1).expand(bsize, npred).contiguous().view(bsize*npred)
    max_y = max_y.view(bsize, 1).expand(bsize, npred).contiguous().view(bsize*npred)

    min_x = torch.clamp(max_x - safe_distance, min=0)
    min_y = np.ceil(crop_w / 2 - width)  # assumes other._width / 2 = self._width / 2
    min_y = torch.tensor(min_y)
    min_y = min_y.view(bsize, 1).expand(bsize, npred).contiguous().view(bsize*npred)

    x_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_h))) * crop_h / 2
    x_filter = x_filter.unsqueeze(0).expand(bsize*npred, crop_h)
    x_filter = torch.min(x_filter, max_x.view(bsize*npred, 1).expand(x_filter.size()))
    x_filter = torch.max(x_filter, min_x.view(bsize*npred, 1))

    x_filter = (x_filter - min_x.view(bsize*npred, 1)) / (max_x - min_x).view(bsize*npred, 1)
    y_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_w))) * crop_w / 2
    y_filter = y_filter.view(1, crop_w).expand(bsize*npred, crop_w)
    y_filter = torch.min(y_filter, max_y.view(bsize*npred, 1))
    y_filter = torch.max(y_filter, min_y.view(bsize*npred, 1))
    y_filter = (y_filter - min_y.view(bsize*npred, 1)) / (max_y.view(bsize*npred, 1) - min_y.view(bsize*npred, 1))
    x_filter = x_filter
    y_filter = y_filter
    proximity_mask = x_filter.view(-1, crop_h, 1) @ y_filter.view(-1, 1, crop_w)
    proximity_mask = proximity_mask.view(bsize, npred, crop_h, crop_w)
    images = images.view(bsize, npred, nchannels, crop_h, crop_w)
    costs, argmax = torch.max((proximity_mask * images[:, :, green_channel].float()).view(bsize, npred, -1), 2)
    if return_argmax:
        argmax = argmax.numpy()[0][0]
        if argmax > 0:
            argmax = argmax/crop_w - crop_h/2, argmax%crop_w - crop_w/2
        else:
            argmax = None

        return costs, proximity_mask, argmax
    else:
        return costs, proximity_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
# data params
    parser.add_argument('-dataset', type=str, default='i80')
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-ncond', type=int, default=20)
    parser.add_argument('-npred', type=int, default=20)
    parser.add_argument('-debug', type=int, default=0)
    opt = parser.parse_args()

    true_costs, new_costs = [], []

    dataloader = DataLoader(None, opt, 'i80')
    inputs, actions, targets, ids, sizes = dataloader.get_batch_fm('train', opt.npred)
    inputs, actions, targets, ids, sizes = dataloader.get_batch_fm('train', opt.npred)
    inputs, actions, targets, ids, sizes = dataloader.get_batch_fm('train', opt.npred)
    inputs, actions, targets, ids, sizes = dataloader.get_batch_fm('train', opt.npred)
    images, states, costs = targets
    states *= (1e-8 + dataloader.s_std.view(1, 1, 4).expand(states.size())).cuda()
    states += dataloader.s_mean.view(1, 1, 4).expand(states.size()).cuda()
    costs = costs[:, :, 0]

    images = Variable(images)
    images.requires_grad=True
    pred_costs, masks = proximity_cost(images, states, sizes)

    plt.plot(costs.view(-1).cpu().numpy())
    plt.plot(pred_costs.data.view(-1).cpu().numpy())
