import torch, numpy, argparse, pdb
from dataloader import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-nshards', type=int, default=1)
parser.add_argument('-T', type=int, default=20)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-lanes', type=int, default=8)
parser.add_argument('-ncond', type=int, default=10)
parser.add_argument('-npred', type=int, default=20)
parser.add_argument('-n_episodes', type=int, default=20)
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
parser.add_argument('-debug', type=int, default=0)

opt = parser.parse_args()


def get_data_stats(targets, costs, min_dist, speed):
    target_images, target_states, target_costs = targets
    bsize = target_images.size(0)
    costs, min_dist, speed = [], [], []
    
    for b in range(bsize):
        for t in range(opt.npred):
            state = target_states[b][t]
            dist = torch.zeros(6).cuda()
            for i in range(6):
                dist[i] = torch.norm(state[0][:2] - state[i+1][:2], 2)
            costs.append(target_costs[b][t][0])
            min_dist.append(torch.min(dist))                
            speed.append(torch.norm(state[0][2:]))
    return costs, min_dist, speed
            

dataloader = DataLoader(None, opt, dataset='i80')
costs, min_dist, speed = [], [], []
for i in range(500):
    inputs, actions, targets = dataloader.get_batch_fm('train')
    costs, min_dist, speed = get_data_stats(targets, costs, min_dist, speed)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(min_dist, speed, costs, s=2)



