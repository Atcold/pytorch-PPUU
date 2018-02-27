import torch
import torch.nn as nn
import torch.legacy.nn as lnn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random, pdb

# Bag-of-cars policy network
class PolicyMLP(nn.Module):
    def __init__(self, opt):
        super(PolicyMLP, self).__init__()
        self.opt = opt

        self.j_network = nn.Sequential(
            nn.BatchNorm1d(opt.n_inputs), 
            nn.Linear(opt.n_inputs, opt.n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(opt.n_hidden), 
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(opt.n_hidden), 
            nn.Linear(opt.n_hidden, opt.n_hidden)
            )

        self.i_network = nn.Sequential(
            nn.BatchNorm1d(2*opt.n_inputs), 
            nn.Linear(2*opt.n_inputs, opt.n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(opt.n_hidden), 
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(opt.n_hidden), 
            nn.Linear(opt.n_hidden, opt.n_hidden)
            )

        self.a_network = nn.Sequential(
            nn.BatchNorm1d(2*opt.ncond*opt.n_hidden), 
            nn.Linear(2*opt.ncond*opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(opt.n_hidden), 
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(opt.n_hidden), 
            nn.Linear(opt.n_hidden, opt.n_actions*opt.npred)
            )

    def forward(self, s, m):
        bsize = s.size(0)
        m = m[:, :, 1:].clone()
        s_j = s[:, :, 0].clone().view(-1, self.opt.ncond, 1, self.opt.n_inputs)
        s_i = s[:, :, 1:]
        s_i = torch.cat((s_j.expand(s_i.size()), s_i), 3)
        s_i = s_i.view(-1, 2*self.opt.n_inputs)
        h_i = self.i_network(s_i)
        h_i = h_i.view(bsize, self.opt.ncond, 6, -1)
        m = m.unsqueeze(3).expand(h_i.size())
        h_i *= m
        h_i = torch.sum(h_i, 2)
        h_j = self.j_network(s_j.view(-1, self.opt.n_inputs))
        h_j = h_j.view(bsize, self.opt.ncond, self.opt.n_hidden)
        h_i = h_i.view(bsize, self.opt.ncond*self.opt.n_hidden)
        h_j = h_j.view(bsize, self.opt.ncond*self.opt.n_hidden)
        a = self.a_network(torch.cat((h_i, h_j), 1))
        a = a.view(bsize, self.opt.npred, self.opt.n_actions)
        return a




