import torch
import torch.nn as nn
import torch.legacy.nn as lnn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random, pdb


# generic model to encode interactions between cars
class BOC(nn.Module):
    def __init__(self, opt):
        super(BOC, self).__init__()
        self.opt = opt

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

    def forward(self, s, m):
        bsize = s.size(0)
        m = m[:, :, 1:].clone()
        s_i = s[:, :, 1:]
        s_j = s[:, :, 0].clone().view(-1, self.opt.ncond, 1, self.opt.n_inputs)
        s_i = torch.cat((s_j.expand(s_i.size()), s_i), 3)
        s_i = s_i.view(-1, 2*self.opt.n_inputs)
        h_i = self.i_network(s_i)
        h_i = h_i.view(bsize, self.opt.ncond, 6, -1)
        m = m.unsqueeze(3).expand(h_i.size())
        h_i *= m
        h_i = torch.sum(h_i, 2)
        h_i = h_i.view(bsize, self.opt.ncond*self.opt.n_hidden)
        return h_i


# Bag-of-cars policy network (deterministic)
class PolicyMLP(nn.Module):
    def __init__(self, opt):
        super(PolicyMLP, self).__init__()
        self.opt = opt

        self.i_model = BOC(opt)

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

    def forward(self, s, m, a):
        bsize = s.size(0)
        h_i = self.i_model(s, m)
        s_j = s[:, :, 0].clone().view(-1, self.opt.ncond, 1, self.opt.n_inputs)
        h_j = self.j_network(s_j.view(-1, self.opt.n_inputs))
        h_j = h_j.view(bsize, self.opt.ncond*self.opt.n_hidden)
        a = self.a_network(torch.cat((h_i, h_j), 1))
        a = a.view(bsize, self.opt.npred, self.opt.n_actions)
        return a



# Bag-of-cars policy network (stochastic)
class PolicyVAE(nn.Module):
    def __init__(self, opt):
        super(PolicyVAE, self).__init__()
        self.opt = opt

        self.i_model = BOC(opt)

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

        self.z_network = nn.Sequential(
            nn.BatchNorm1d(opt.n_actions*opt.npred), 
            nn.Linear(opt.n_actions*opt.npred, opt.n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(opt.n_hidden), 
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(opt.n_hidden), 
            nn.Linear(opt.n_hidden, 2*opt.nz)
            )

        self.a_network = nn.Sequential(
            nn.BatchNorm1d(2*opt.ncond*opt.n_hidden + opt.nz), 
            nn.Linear(2*opt.ncond*opt.n_hidden + opt.nz, opt.n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(opt.n_hidden), 
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(opt.n_hidden), 
            nn.Linear(opt.n_hidden, opt.n_actions*opt.npred)
            )


    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, a):
        bsize = a.size(0)
        z_params = self.z_network(a.view(bsize, -1)).view(bsize, self.opt.nz, 2)
        mu = z_params[:, :, 0]
        logvar = z_params[:, :, 1]
        return mu, logvar


    def forward(self, s, m, a):
        bsize = s.size(0)
        h_i = self.i_model(s, m)
        mu, logvar = self.encode(a)
        z = self.reparameterize(mu, logvar)
        s_j = s[:, :, 0].clone().view(-1, self.opt.ncond, 1, self.opt.n_inputs)
        h_j = self.j_network(s_j.view(-1, self.opt.n_inputs))
        h_j = h_j.view(bsize, self.opt.ncond*self.opt.n_hidden)
        a = self.a_network(torch.cat((h_i, h_j, z), 1))
        a = a.view(bsize, self.opt.npred, self.opt.n_actions)
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return a, KL






        

        



