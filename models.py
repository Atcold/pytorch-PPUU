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











class FwdCNN(nn.Module):
    def __init__(self, opt):
        super(FwdCNN, self).__init__()
        self.opt = opt

        self.f_encoder = nn.Sequential(
            nn.Conv2d(3*opt.ncond, opt.nfeature, 4, 2, 1), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1), 
            nn.BatchNorm2d(opt.nfeature)
        )


        if self.opt.tie_action == 1:
            self.aemb_size = opt.nfeature
        else:
            self.aemb_size = opt.nfeature*12*2

        self.action_embed = nn.Sequential(
            nn.BatchNorm1d(opt.n_actions), 
            nn.Linear(opt.n_actions, opt.nfeature), 
            nn.BatchNorm1d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.Linear(opt.nfeature, opt.nfeature), 
            nn.BatchNorm1d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.Linear(opt.nfeature, self.aemb_size)
        )

        self.f_decoder = nn.Sequential(
            nn.ConvTranspose2d(opt.nfeature, opt.nfeature, (4, 5), 2, 1), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.ConvTranspose2d(opt.nfeature, opt.nfeature, (5, 5), 2, (1, 1)), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.ConvTranspose2d(opt.nfeature, 3, (2, 2), 2, (0, 1))
        )


    def forward(self, inputs, actions, target):
        bsize = inputs.size(0)
        inputs = inputs.view(bsize, self.opt.ncond, 3, 97, 20)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)
        pred = []
        for t in range(npred):
            h = self.f_encoder(inputs.view(bsize, self.opt.ncond*3, 97, 20))
            a = self.action_embed(actions[:, t].contiguous())
            if self.opt.tie_action == 1:
                h = h + a.view(bsize, self.opt.nfeature, 1, 1).expand(h.size())
            else:
                h = h + a.view(bsize, self.opt.nfeature, 12, 2)
            out = self.f_decoder(h)[:, :, :-1].clone()
            out = out.view(bsize, 1, 3, 97, 20)
            out = out + inputs[:, -1].unsqueeze(1).clone()
            if self.opt.sigmout == 1:
                out = F.sigmoid(out)
            pred.append(out)
            inputs = torch.cat((inputs[:, 1:], out), 1)

        pred = torch.cat(pred, 1)
        return pred, Variable(torch.zeros(1))


    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()




class FwdCNNJoint(nn.Module):
    def __init__(self, opt):
        super(FwdCNN, self).__init__()
        self.opt = opt

        self.f_encoder = nn.Sequential(
            nn.Conv2d(3*opt.ncond, opt.nfeature, 4, 2, 1), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1), 
            nn.BatchNorm2d(opt.nfeature)
        )

        self.action_embed = nn.Sequential(
            nn.Linear(opt.n_actions*opt.npred, opt.nfeature), 
            nn.BatchNorm1d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.Linear(opt.nfeature, opt.nfeature), 
            nn.BatchNorm1d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.Linear(opt.nfeature, opt.nfeature)
        )

        self.f_decoder = nn.Sequential(
            nn.ConvTranspose2d(opt.nfeature, opt.nfeature, (4, 5), 2, 1), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.ConvTranspose2d(opt.nfeature, opt.nfeature, (5, 5), 2, (1, 1)), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.ConvTranspose2d(opt.nfeature, 3*opt.npred, (2, 2), 2, (0, 1))
        )


    def forward(self, inputs, actions, target):
        bsize = inputs.size(0)
        inputs = inputs.view(bsize, self.opt.ncond*3, 97, 20)
        actions = actions.view(bsize, self.opt.npred*self.opt.n_actions)
        h = self.f_encoder(inputs)
        a = self.action_embed(actions)
        h = h + a.view(bsize, self.opt.nfeature, 1, 1).expand(h.size())
        out = self.f_decoder(h)[:, :, :-1].clone()
        out = out.view(bsize, self.opt.npred, 3, 97, 20)
        inputs = inputs.view(bsize, self.opt.ncond, 3, 97, 20)
        last_input = inputs[:, -1].clone().view(bsize, 1, 3, 97, 20)
        pred = out + last_input.expand(out.size())
        return pred, Variable(torch.zeros(1))


    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()
        







class PolicyCNN(nn.Module):
    def __init__(self, opt):
        super(PolicyCNN, self).__init__()
        self.opt = opt

        self.convnet = nn.Sequential(
            nn.Conv2d(3*opt.ncond, opt.nfeature, 4, 2, 1), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.ReLU(), 
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.ReLU(), 
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.ReLU()
        )

        self.embed = nn.Sequential(
            nn.BatchNorm1d(opt.ncond*opt.n_inputs), 
            nn.Linear(opt.ncond*opt.n_inputs, opt.n_hidden), 
            nn.BatchNorm1d(opt.n_hidden), 
            nn.ReLU(), 
            nn.Linear(opt.n_hidden, opt.n_hidden), 
            nn.BatchNorm1d(opt.n_hidden)
        )

        self.hsize = opt.nfeature*12*2
        self.fc = nn.Sequential(
            nn.Linear(self.hsize + opt.n_hidden, opt.n_hidden), 
            nn.ReLU(), 
            nn.Linear(opt.n_hidden, opt.n_hidden), 
            nn.ReLU(), 
            nn.Linear(opt.n_hidden, opt.npred*opt.n_actions)
        )

    def forward(self, state_images, states, actions):
        bsize = state_images.size(0)
        state_images = state_images.view(bsize, 3*self.opt.ncond, 97, 20)
        states = states.view(bsize, -1)
        hi = self.convnet(state_images).view(bsize, self.hsize)
        hs = self.embed(states)
        a = self.fc(torch.cat((hi, hs), 1))
        a = a.view(bsize, self.opt.npred, self.opt.n_actions)
        return a, Variable(torch.zeros(1))


    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()









class PolicyCNN_VAE(nn.Module):
    def __init__(self, opt):
        super(PolicyCNN_VAE, self).__init__()
        self.opt = opt

        self.convnet = nn.Sequential(
            nn.Conv2d(3*opt.ncond, opt.nfeature, 4, 2, 1), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.ReLU(), 
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.ReLU(), 
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.ReLU()
        )

        self.embed = nn.Sequential(
            nn.BatchNorm1d(opt.ncond*opt.n_inputs), 
            nn.Linear(opt.ncond*opt.n_inputs, opt.n_hidden), 
            nn.BatchNorm1d(opt.n_hidden), 
            nn.ReLU(), 
            nn.Linear(opt.n_hidden, opt.n_hidden), 
            nn.BatchNorm1d(opt.n_hidden)
        )
        self.hsize = opt.nfeature*12*2

        self.fc = nn.Sequential(
            nn.Linear(self.hsize + opt.n_hidden + opt.nz, opt.n_hidden), 
            nn.BatchNorm1d(opt.n_hidden),
            nn.ReLU(), 
            nn.Linear(opt.n_hidden, opt.n_hidden), 
            nn.BatchNorm1d(opt.n_hidden),
            nn.ReLU(), 
            nn.Linear(opt.n_hidden, opt.npred*opt.n_actions)
        )

        self.z_network = nn.Sequential(
            nn.BatchNorm1d(opt.n_actions*opt.npred + self.hsize + opt.n_hidden),
            nn.Linear(opt.n_actions*opt.npred + self.hsize + opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(opt.n_hidden),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(opt.n_hidden),
            nn.Linear(opt.n_hidden, 2*opt.nz)
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


    def forward(self, state_images, states, actions):
        bsize = state_images.size(0)
        state_images = state_images.view(bsize, 3*self.opt.ncond, 97, 20)
        states = states.view(bsize, -1)
        actions = actions.view(bsize, -1)
        hi = self.convnet(state_images).view(bsize, self.hsize)
        hs = self.embed(states)

        h = torch.cat((hi, hs), 1)
        mu, logvar = self.encode(torch.cat((h, actions), 1))
        z = self.reparameterize(mu, logvar)

        a = self.fc(torch.cat((h, z), 1))
        a = a.view(bsize, self.opt.npred, self.opt.n_actions)
        return a, Variable(torch.zeros(1))


    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()































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
        return a, Variable(torch.zeros(1))


    def intype(self, t):
        if t == 'cpu':
            self.i_model.cpu()
            self.i_model.i_network.cpu()
            self.j_network.cpu()
            self.a_network.cpu()
        elif t == 'gpu':
            self.i_model.cuda()
            self.i_model.i_network.cuda()
            self.j_network.cuda()
            self.a_network.cuda()



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

    def intype(self, t):
        if t == 'cpu':
            self.i_model.cpu()
            self.i_model.i_network.cpu()
            self.j_network.cpu()
            self.a_network.cpu()
            self.z_network.cpu()
        elif t == 'gpu':
            self.i_model.cuda()
            self.i_model.i_network.cuda()
            self.j_network.cuda()
            self.a_network.cuda()
            self.z_network.cuda()











