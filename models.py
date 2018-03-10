import torch
import torch.nn as nn
import torch.legacy.nn as lnn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random, pdb

####################
# Basic modules
####################

# encodes a sequence of input frames and an action to a hidden representation
class encoder(nn.Module):
    def __init__(self, opt, a_size):
        super(encoder, self).__init__()
        self.opt = opt
        self.a_size = a_size
        # frame encoder
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

        # action encoder
        self.a_encoder = nn.Sequential(
            nn.BatchNorm1d(a_size), 
            nn.Linear(a_size, opt.nfeature), 
            nn.BatchNorm1d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.Linear(opt.nfeature, opt.nfeature), 
            nn.BatchNorm1d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.Linear(opt.nfeature, self.aemb_size)
        )

    def forward(self, inputs, actions):
        bsize = inputs.size(0)
        h = self.f_encoder(inputs.view(bsize, self.opt.ncond*3, self.opt.height, self.opt.width))
        a = self.a_encoder(actions.contiguous().view(bsize, self.a_size))
        if self.opt.tie_action == 1:
            h = h + a.view(bsize, self.opt.nfeature, 1, 1).expand(h.size())
        else:
            h = h + a.view(bsize, self.opt.nfeature, 12, 2)
        return h

        
# decodes a hidden state into a predicted frame
class decoder(nn.Module):
    def __init__(self, opt):
        super(decoder, self).__init__()
        self.opt = opt
        self.f_decoder = nn.Sequential(
            nn.ConvTranspose2d(opt.nfeature, opt.nfeature, (4, 5), 2, 1), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.ConvTranspose2d(opt.nfeature, opt.nfeature, (5, 5), 2, (1, 1)), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.ConvTranspose2d(opt.nfeature, 3, (2, 2), 2, (0, 1))
        )

    def forward(self, h):
        bsize = h.size(0)
        out = self.f_decoder(h)[:, :, :-1].clone()
        return out.view(bsize, 1, 3, self.opt.height, self.opt.width)


# decodes a hidden state into a sequences of predicted frames, using temporal deconvolution
class decoder_deconv(nn.Module):
    def __init__(self, opt):
        super(decoder_deconv, self).__init__()
        self.opt = opt
        assert(opt.npred == 50, 'hardcoded for npred=50')

        self.f_decoder1 = nn.Sequential(
            nn.ConvTranspose3d(opt.nfeature, opt.nfeature, (4, 4, 5), stride=2, padding=(0, 1, 1)), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.ConvTranspose3d(opt.nfeature, opt.nfeature, (4, 5, 5), stride=2, padding=(0, 1, 1)), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.ConvTranspose3d(opt.nfeature, opt.nfeature, (4, 2, 2), stride=2, padding=(0, 0, 1), output_padding=(1, 0, 0)), 
            nn.LeakyReLU(0.2)
        )

        self.f_decoder2 = nn.Sequential(
            nn.ConvTranspose3d(opt.nfeature, 3, (6, 1, 1), stride=(2, 1, 1))
        )
        

    def forward(self, h):
        bsize = h.size(0)
        h = h.unsqueeze(2)
        out = self.f_decoder2(self.f_decoder1(h))[:, :, :, :-1].clone()
        out = out.permute(0, 2, 1, 3, 4)
        return out



# encodes a sequence of frames or errors and produces a latent variable
class z_network(nn.Module):
    def __init__(self, opt, n_inputs):
        super(z_network, self).__init__()
        self.opt = opt
        self.n_inputs = n_inputs

        self.conv = nn.Sequential(
            nn.Conv2d(3*n_inputs, opt.nfeature, 4, 2, 1), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1), 
            nn.BatchNorm2d(opt.nfeature)
        )

        self.fc = nn.Sequential(
            nn.Linear(opt.nfeature*12*2, opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.Linear(opt.nfeature, opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.Linear(opt.nfeature, 2*opt.nz)
        )
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, inputs):
        bsize = inputs.size(0)
        inputs = inputs.view(bsize, self.n_inputs*3, self.opt.height, self.opt.width)
        z_params = self.fc(self.conv(inputs).view(bsize, -1)).view(bsize, self.opt.nz, 2)
        mu = z_params[:, :, 0]
        logvar = z_params[:, :, 1]
        return mu, logvar

    def forward(self, inputs):
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


# expands a latent variable into a sequence of variables, one for each frame
class z_expander(nn.Module):
    def __init__(self, opt):
        super(z_expander, self).__init__()
        self.opt = opt
        self.z_expander = nn.Sequential(
            nn.Linear(opt.nz, opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.Linear(opt.nfeature, opt.nfeature), 
            nn.LeakyReLU(0.2), 
            nn.Linear(opt.nfeature, opt.npred * opt.nfeature * 12 * 2)
        )

    def forward(self, z):
        bsize = z.size(0)
        z_exp = self.z_expander(z).view(bsize, self.opt.npred, self.opt.nfeature, 12, 2)
        return z_exp


###############
# Main models
###############

# forward model, deterministic

class FwdCNN(nn.Module):
    def __init__(self, opt):
        super(FwdCNN, self).__init__()
        self.opt = opt
        self.encoder = encoder(opt, opt.n_actions)
        self.decoder = decoder(opt)

    def forward(self, inputs, actions, target):
        bsize = inputs.size(0)
        inputs = inputs.view(bsize, self.opt.ncond, 3, self.opt.height, self.opt.width)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)
        pred = []
        for t in range(npred):
            h = self.encoder(inputs, actions[:, t])
            pred_ = F.sigmoid(self.decoder(h) + inputs[:, -1].unsqueeze(1).clone())
            pred.append(pred_)
            inputs = torch.cat((inputs[:, 1:], pred_), 1)

        pred = torch.cat(pred, 1)
        return pred, Variable(torch.zeros(1))


    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()


# forward model with a fixed prior
class FwdCNN_VAE_FP(nn.Module):
    def __init__(self, opt):
        super(FwdCNN_VAE_FP, self).__init__()
        self.opt = opt
        self.encoder = encoder(opt, opt.n_actions)
        self.decoder = decoder(opt)
        self.z_network = z_network(opt, opt.ncond+opt.npred)
        self.z_expander = z_expander(opt)

    def forward(self, inputs, actions, targets):
        bsize = inputs.size(0)
        inputs = inputs.view(bsize, self.opt.ncond, 3, self.opt.height, self.opt.width)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)

        z, mu, logvar = self.z_network(torch.cat((inputs, targets), 1))
        z_exp = self.z_expander(z)

        pred = []
        for t in range(npred):
            h = self.encoder(inputs, actions[:, t])
            h = h + z_exp[:, t]
            pred_ = F.sigmoid(self.decoder(h) + inputs[:, -1].unsqueeze(1).clone())
            pred.append(pred_)
            inputs = torch.cat((inputs[:, 1:], pred_), 1)

        pred = torch.cat(pred, 1)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld /= bsize
        return pred, kld

    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()


# forward model with a learned prior
class FwdCNN_VAE_LP(nn.Module):
    def __init__(self, opt):
        super(FwdCNN_VAE_LP, self).__init__()
        self.opt = opt
        self.encoder = encoder(opt, opt.n_actions)
        self.decoder = decoder(opt)
        self.z_network = z_network(opt, opt.ncond+opt.npred)
        self.q_network = z_network(opt, opt.ncond)
        self.z_expander = z_expander(opt)

    def forward(self, inputs, actions, targets):
        bsize = inputs.size(0)
        inputs = inputs.view(bsize, self.opt.ncond, 3, self.opt.height, self.opt.width)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)

        z, mu1, logvar1 = self.z_network(torch.cat((inputs, targets), 1))
        _, mu2, logvar2 = self.q_network(inputs)
        z_exp = self.z_expander(z)

        pred = []
        for t in range(npred):
            h = self.encoder(inputs, actions[:, t])
            h = h + z_exp[:, t]
            pred_ = F.sigmoid(self.decoder(h) + inputs[:, -1].unsqueeze(1).clone())
            pred.append(pred_)
            inputs = torch.cat((inputs[:, 1:], pred_), 1)

        pred = torch.cat(pred, 1)

        sigma1 = logvar1.mul(0.5).exp() 
        sigma2 = logvar2.mul(0.5).exp() 
        kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
        kld = torch.sum(kld) / bsize
        return pred, kld

    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()







# forward model with a learned prior
class FwdCNN_EEN_LP(nn.Module):
    def __init__(self, opt):
        super(FwdCNN_EEN_LP, self).__init__()
        self.opt = opt
        self.encoder = encoder(opt, opt.n_actions)
        self.decoder = decoder(opt)
        self.z_network = z_network(opt, opt.ncond+opt.npred)
        self.q_network = z_network(opt, opt.ncond)
        self.z_expander = z_expander(opt)

    def forward(self, inputs, actions, targets):
        bsize = inputs.size(0)
        inputs = inputs.view(bsize, self.opt.ncond, 3, self.opt.height, self.opt.width)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)

        # deterministic pass
        pred = []
        for t in range(npred):
            h = self.encoder(inputs, actions[:, t])
            h = h
            pred_ = F.sigmoid(self.decoder(h) + inputs[:, -1].unsqueeze(1).clone())
            pred.append(pred_)
            inputs = torch.cat((inputs[:, 1:], pred_), 1)

        pred1 = torch.cat(pred, 1)
        errors = targets - pred1

        z, mu1, logvar1 = self.z_network(torch.cat((inputs, errors), 1))
        _, mu2, logvar2 = self.q_network(inputs)
        z_exp = self.z_expander(z)

        # latent pass
        pred = []
        for t in range(npred):
            h = self.encoder(inputs, actions[:, t])
            h = h + z_exp[:, t]
            pred_ = F.sigmoid(self.decoder(h) + inputs[:, -1].unsqueeze(1).clone())
            pred.append(pred_)
            inputs = torch.cat((inputs[:, 1:], pred_), 1)

        pred2 = torch.cat(pred, 1)

        sigma1 = logvar1.mul(0.5).exp() 
        sigma2 = logvar2.mul(0.5).exp() 
        kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
        kld = torch.sum(kld) / bsize
        return pred1, pred2, kld

    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()







###########################################################
# Models with convolutional decoders (not autoregressive)
###########################################################

# forward model, deterministic

class FwdCNN2(nn.Module):
    def __init__(self, opt):
        super(FwdCNN2, self).__init__()
        self.opt = opt
        self.encoder = encoder(opt, opt.n_actions*opt.npred)
        self.decoder = decoder_deconv(opt)

    def forward(self, inputs, actions, target):
        bsize = inputs.size(0)
        inputs = inputs.view(bsize, self.opt.ncond, 3, self.opt.height, self.opt.width)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)
        pred = []
        h = self.encoder(inputs, actions)
        pred = self.decoder(h)
        pred = F.sigmoid(pred + inputs[:, -1].unsqueeze(1).expand(pred.size()))
        
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
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return a, kld


    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()
