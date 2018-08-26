import torch
import torch.nn as nn
import torch.legacy.nn as lnn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import random, pdb, copy, os, math, numpy, copy, time
import utils



####################
# Basic modules
####################

# encodes a sequence of input frames and states, and optionally a cost or action, to a hidden representation
class encoder(nn.Module):
    def __init__(self, opt, a_size, n_inputs, states=True, state_input_size=4):
        super(encoder, self).__init__()
        self.opt = opt
        self.a_size = a_size
        self.n_inputs = opt.ncond if n_inputs is None else n_inputs
        # frame encoder
        if opt.layers == 3:
            if self.opt.fmap_geom == 0:
                self.feature_maps = [opt.nfeature, opt.nfeature, opt.nfeature]
            elif self.opt.fmap_geom == 1:
                assert(opt.nfeature % 4 == 0)
                self.feature_maps = [int(opt.nfeature/4), int(opt.nfeature/2), opt.nfeature]

            self.f_encoder = nn.Sequential(
                nn.Conv2d(3*self.n_inputs, self.feature_maps[0], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[0], self.feature_maps[1], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[1], self.feature_maps[2], 4, 2, 1),
            )
        elif opt.layers == 4:
            if self.opt.fmap_geom == 0:
                self.feature_maps = [opt.nfeature, opt.nfeature, opt.nfeature, opt.nfeature]
            elif self.opt.fmap_geom == 1:
                assert(opt.nfeature % 8 == 0)
                self.feature_maps = [int(opt.nfeature/8), int(opt.nfeature/4), int(opt.nfeature/2), opt.nfeature]
            self.f_encoder = nn.Sequential(
                nn.Conv2d(3*self.n_inputs, self.feature_maps[0], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[0], self.feature_maps[1], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[1], self.feature_maps[2], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[2], self.feature_maps[3], 4, 2, 1)
            )


        if states:
            n_hidden = self.feature_maps[-1]
            # state encoder
            self.s_encoder = nn.Sequential(
                nn.Linear(state_input_size*self.n_inputs, n_hidden),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(n_hidden, n_hidden),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(n_hidden, self.opt.hidden_size)
            )

        if a_size > 0:
            # action or cost encoder
            n_hidden = self.feature_maps[-1]
            self.a_encoder = nn.Sequential(
                nn.Linear(a_size, n_hidden),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(n_hidden, n_hidden),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(n_hidden, self.opt.hidden_size)
            )

    def forward(self, images, states=None, actions=None):
        bsize = images.size(0)
        h = self.f_encoder(images.view(bsize, self.n_inputs*3, self.opt.height, self.opt.width))
        if states is not None:
            h = utils.combine(h, self.s_encoder(states.contiguous().view(bsize, -1)).view(h.size()), self.opt.combine)
        if actions is not None:
            a = self.a_encoder(actions.contiguous().view(bsize, self.a_size))
            h = utils.combine(h, a.view(h.size()), self.opt.combine)
        return h



class u_network(nn.Module):
    def __init__(self, opt):
        super(u_network, self).__init__()
        self.opt = opt
        self.encoder = nn.Sequential(
            nn.Conv2d(self.opt.nfeature, self.opt.nfeature, 4, 2, 1), 
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv2d(self.opt.nfeature, self.opt.nfeature, (4, 1), 2, 1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.opt.nfeature, self.opt.nfeature, (4, 1), 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.opt.nfeature, self.opt.nfeature, (4, 3), 2, 0)
        )

        assert(self.opt.layers == 3) # hardcoded sizes
        self.hidden_size = self.opt.nfeature*3*2
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.opt.nfeature), 
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Linear(self.opt.nfeature, self.hidden_size)
        )
    
    def forward(self, h):
        h1 = self.encoder(h)
        h2 = self.fc(h1.view(-1, self.hidden_size))
        h2 = h2.view(h1.size())
        h3 = self.decoder(h2)
        return h


# decodes a hidden state into a predicted frame, a predicted state and a predicted cost vector
class decoder(nn.Module):
    def __init__(self, opt, n_out=1):
        super(decoder, self).__init__()
        self.opt = opt
        self.n_out = n_out
        if self.opt.layers == 3:
            if self.opt.fmap_geom == 0:
                self.feature_maps = [opt.nfeature, opt.nfeature, opt.nfeature]
            elif self.opt.fmap_geom == 1:
                assert(opt.nfeature % 4 == 0)
                self.feature_maps = [int(opt.nfeature/4), int(opt.nfeature/2), opt.nfeature]

            self.f_decoder = nn.Sequential(
                nn.ConvTranspose2d(self.feature_maps[2], self.feature_maps[1], (4, 4), 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(self.feature_maps[1], self.feature_maps[0], (5, 5), 2, (0, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(self.feature_maps[0], self.n_out*3, (2, 2), 2, (0, 1))
            )

            self.h_reducer = nn.Sequential(
                nn.Conv2d(self.feature_maps[2], self.feature_maps[2], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[2], self.feature_maps[2], (4, 1), (2, 1), 0),
                nn.LeakyReLU(0.2, inplace=True)
            )

        elif self.opt.layers == 4:
            if self.opt.fmap_geom == 0:
                self.feature_maps = [opt.nfeature, opt.nfeature, opt.nfeature, opt.nfeature]
            elif self.opt.fmap_geom == 1:
                assert(opt.nfeature % 8 == 0)
                self.feature_maps = [int(opt.nfeature/8), int(opt.nfeature/4), int(opt.nfeature/2), opt.nfeature]

            self.f_decoder = nn.Sequential(
                nn.ConvTranspose2d(self.feature_maps[3], self.feature_maps[2], (4, 4), 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(self.feature_maps[2], self.feature_maps[1], (5, 5), 2, (0, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(self.feature_maps[1], self.feature_maps[0], (2, 4), 2, (1, 0)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(self.feature_maps[0], self.n_out*3, (2, 2), 2, (1, 0))
            )

            self.h_reducer = nn.Sequential(
                nn.Conv2d(opt.nfeature, opt.nfeature, (4, 1), (2, 1), 0),
                nn.LeakyReLU(0.2, inplace=True)
            )

        
        n_hidden = self.feature_maps[-1]
        self.c_predictor = nn.Sequential(
            nn.Linear(2*n_hidden, n_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_hidden, self.n_out*2),
            nn.Sigmoid()
        )

        self.s_predictor = nn.Sequential(
            nn.Linear(2*n_hidden, n_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_hidden, self.n_out*4)
        )


    def forward(self, h):
        bsize = h.size(0)
        h = h.view(bsize, self.feature_maps[-1], self.opt.h_height, self.opt.h_width)
        h_reduced = self.h_reducer(h).view(bsize, -1)
        pred_cost = self.c_predictor(h_reduced)
        pred_state = self.s_predictor(h_reduced)
        pred_image = self.f_decoder(h)
        pred_image = pred_image[:, :, :self.opt.height, :self.opt.width].clone()
        pred_image = pred_image.view(bsize, 1, 3*self.n_out, self.opt.height, self.opt.width)
        return pred_image, pred_state, pred_cost





# expands a latent variable to the size of the hidden representation
class z_expander(nn.Module):
    def __init__(self, opt, n_steps):
        super(z_expander, self).__init__()
        self.opt = opt
        self.n_steps = n_steps
        self.z_expander = nn.Sequential(
            nn.Linear(opt.nz, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, n_steps * opt.nfeature * self.opt.h_height * self.opt.h_width)
        )

    def forward(self, z):
        bsize = z.size(0)
        z_exp = self.z_expander(z).view(bsize, self.n_steps, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
        return z_exp



# maps a hidden representation to a distribution over latent variables. 
# We use this for the VAE models. 
class z_network_gaussian(nn.Module):
    def __init__(self, opt):
        super(z_network_gaussian, self).__init__()
        self.opt = opt

        self.network = nn.Sequential(
            nn.Linear(opt.nfeature*self.opt.h_height*self.opt.h_width, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, 2*opt.nz)
        )

    def reparameterize(self, mu, logvar, sample):
        if self.training or sample:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, inputs):
        bsize = inputs.size(0)
        inputs = inputs.view(bsize, self.opt.nfeature*self.opt.h_height*self.opt.h_width)
        z_params = self.network(inputs).view(-1, self.opt.nz, 2)
        mu = z_params[:, :, 0]
        logvar = z_params[:, :, 1]
        return mu, logvar

    def forward(self, inputs, sample=True):
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar, sample)
        return z, mu, logvar



class action_indep_net(nn.Module):
    def __init__(self, opt):
        super(action_indep_net, self).__init__()
        self.opt = opt
        self.encoder = encoder(opt, 0, opt.ncond)

        self.network_s = nn.Sequential(
            nn.Linear(opt.nfeature*self.opt.h_height*self.opt.h_width, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, 2*opt.nz)
        )

        self.network_sa = nn.Sequential(
            nn.Linear(opt.nfeature*self.opt.h_height*self.opt.h_width, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, 2*opt.nz)
        )

        self.a_encoder = nn.Sequential(
            nn.Linear(opt.n_actions, self.opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.opt.hidden_size)
        )

    def forward(self, input_images, input_states, actions, z):
        bsize = input_images.size(0)
        h = self.encoder(input_images, input_states).view(bsize, self.opt.hidden_size)
        a_emb = self.a_encoder(actions).view(h.size())
        z1 = self.network_s(h).view(bsize, self.opt.nz, 2)
        z2 = self.network_sa(h + a_emb).view(bsize, self.opt.nz, 2)
        mu1 = z1[:, :, 0]
        logvar1 = z1[:, :, 1]
        mu2 = z2[:, :, 0]
        logvar2 = z2[:, :, 1]
        sigma1 = logvar1.mul(0.5).exp_()
        sigma2 = logvar1.mul(0.5).exp_()
        loss1 = utils.log_pdf(z, mu1, sigma1)
        loss2 = utils.log_pdf(z, mu2, sigma2)
        return loss1, loss2










# takes as input a sequence of frames, states and actions, and outputs the parameters of a 
# Gaussian Mixture Model. 
class PriorMDN(nn.Module):
    def __init__(self, opt):
        super(PriorMDN, self).__init__()
        self.opt = opt
        self.n_inputs = opt.ncond
        self.encoder = encoder(opt, opt.n_actions, opt.ncond)

        self.network = nn.Sequential(
            nn.Linear(self.opt.hidden_size, opt.n_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.pi_net = nn.Linear(opt.n_hidden, opt.n_mixture)
        self.mu_net = nn.Linear(opt.n_hidden, opt.n_mixture*opt.nz)
        self.sigma_net = nn.Linear(opt.n_hidden, opt.n_mixture*opt.nz)


    def forward(self, input_images, input_states, input_actions):
        bsize = input_images.size(0)
        h = self.encoder(input_images, input_states, input_actions).view(bsize, -1)
        h = self.network(h)
        pi = F.softmax(self.pi_net(h), dim=1)
        mu = self.mu_net(h).view(bsize, self.opt.n_mixture, self.opt.nz)
        sigma = F.softplus(self.sigma_net(h)).view(bsize, self.opt.n_mixture, self.opt.nz)
        sigma = torch.clamp(sigma, min=1e-3)
        return pi, mu, sigma


    # first extract z vectors by passing inputs, actions and targets through an external model, and uses these as targets. 
    # Useful for training the prior network to predict the z vectors inferred by a previously trained forward model. 
    def forward_thru_model(self, model, inputs, actions, targets):
        input_images, input_states = inputs
        bsize = input_images.size(0)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)
        ploss = Variable(torch.zeros(1).cuda())
    
        for t in range(npred):
            h_x = model.encoder(input_images, input_states, actions[:, t])
            target_images, target_states, target_costs = targets
            h_y = model.y_encoder(torch.cat((input_images, target_images[:, t].unsqueeze(1).contiguous()), 1), input_states, actions[:, t])
            z = model.z_network(utils.combine(h_x, h_y, model.opt.combine).view(bsize, -1))
            pi, mu, sigma = self(input_images, input_states, actions[:, t])
            # prior loss
            ploss = utils.mdn_loss_fn(pi, sigma, mu, z)
            z_exp = model.z_expander(z).view(bsize, model.opt.nfeature, model.opt.h_height, model.opt.h_width)
            h_x = h_x.view(bsize, model.opt.nfeature, model.opt.h_height, model.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), model.opt.combine)
            pred_image, pred_state, pred_cost = model.decoder(h)
            pred_image.detach()
            pred_state.detach()
            pred_cost.detach()
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)

        return ploss


# takes as input a sequence of frames, states and actions, and outputs the parameters of a 
# Gaussian Mixture Model. 
class PriorGaussian(nn.Module):
    def __init__(self, opt, nz):
        super(PriorGaussian, self).__init__()
        self.opt = opt
        self.n_inputs = opt.ncond
        self.encoder = encoder(opt, 0, opt.ncond)
        self.nz = nz

        self.network = nn.Sequential(
            nn.Linear(self.opt.hidden_size, opt.n_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mu_net = nn.Linear(opt.n_hidden, nz)
        self.sigma_net = nn.Linear(opt.n_hidden, nz)


    def forward(self, input_images, input_states, normalize_inputs=False, normalize_outputs=False, n_samples=1):
        if normalize_inputs:
            input_images = input_images.clone().float().div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = Variable(input_images.cuda()).unsqueeze(0)
            input_states = Variable(input_states.cuda()).unsqueeze(0)

        bsize = input_images.size(0)
        h = self.encoder(input_images, input_states).view(bsize, -1)
        h = self.network(h)
        mu = self.mu_net(h).view(bsize, self.nz)
        sigma = F.softplus(self.sigma_net(h)).view(bsize, self.nz)
        sigma = torch.clamp(sigma, min=1e-3)

        eps = Variable(torch.randn(bsize, n_samples, self.opt.n_actions).cuda())
        a = eps * sigma.view(bsize, 1, self.opt.n_actions)
        a = a + mu.view(bsize, 1, self.opt.n_actions)


        if normalize_outputs:
            a = a.data
            a *= self.stats['a_std'].view(1, 1, 2).expand(a.size()).cuda()
            a += self.stats['a_mean'].view(1, 1, 2).expand(a.size()).cuda()

        return mu, sigma, a














# Mixture Density network (fully-connected). 
class v_network_mdn_fc(nn.Module):
    def __init__(self, opt, n_outputs):
        super(v_network_mdn_fc, self).__init__()
        self.opt = opt
        self.n_outputs = n_outputs

        self.network = nn.Sequential(
            nn.Linear(self.opt.hidden_size, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.pi_net = nn.Linear(opt.nfeature, opt.n_mixture)
        self.mu_net = nn.Linear(opt.nfeature, opt.n_mixture*n_outputs)
        self.sigma_net = nn.Linear(opt.nfeature, opt.n_mixture*n_outputs)

    def forward(self, h):
        bsize = h.size(0)
        h = h.view(bsize, self.opt.hidden_size)
        h = self.network(h)
        pi = F.softmax(self.pi_net(h), dim=1)
        mu = self.mu_net(h).view(bsize, self.opt.n_mixture, self.n_outputs)
        sigma = F.softplus(self.sigma_net(h)).view(bsize, self.opt.n_mixture, self.n_outputs)
        sigma = torch.clamp(sigma, min=1e-3)
        return pi, mu, sigma



class v_network(nn.Module):
    def __init__(self, opt):
        super(v_network, self).__init__()
        self.opt = opt

        self.network = nn.Sequential(
            nn.Linear(self.opt.hidden_size, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Linear(opt.nfeature, opt.nz)
        )

    def forward(self, h):
        bsize = h.size(0)
        h = h.view(bsize, self.opt.hidden_size)
        u = self.network(h)
        u = u / torch.norm(u, 2, 1).unsqueeze(1)
        return u 







# combines a sequence of images with the state vector. 
class policy_encoder(nn.Module):
    def __init__(self, opt):
        super(policy_encoder, self).__init__()
        self.opt = opt

        self.convnet = nn.Sequential(
            nn.Conv2d(3*opt.ncond, opt.nfeature, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1),
            nn.ReLU()
        )

        self.embed = nn.Sequential(
            nn.Linear(opt.ncond*opt.n_inputs, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden)
        )

        self.hsize = opt.nfeature*self.opt.h_height*self.opt.h_width

    def forward(self, state_images, states):
        bsize = state_images.size(0)
        state_images = state_images.view(bsize, 3*self.opt.ncond, self.opt.height, self.opt.width)
        states = states.view(bsize, -1)
        hi = self.convnet(state_images).view(bsize, self.hsize)
        hs = self.embed(states)
        h = torch.cat((hi, hs), 1)
        return h


    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()







###############
# Main models
###############

# forward model, deterministic
class FwdCNN(nn.Module):
    def __init__(self, opt, mfile):
        super(FwdCNN, self).__init__()
        self.opt = opt
        # If we are given a model file, use it to initialize this model. 
        # otherwise initialize from scratch
        if mfile == '':
            self.encoder = encoder(opt, opt.n_actions, opt.ncond)
            self.decoder = decoder(opt)
        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.encoder.n_inputs = opt.ncond

    def forward(self, inputs, actions, target, sampling=None, p_dropout=None):
        t0 = time.time()
        npred = actions.size(1)
        input_images, input_states = inputs
        pred_images, pred_states, pred_costs = [], [], []
        for t in range(npred):
            h = self.encoder(input_images, input_states, actions[:, t])
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        print(time.time() - t0)
        return [pred_images, pred_states, pred_costs], Variable(torch.zeros(1).cuda())

    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()


# forward model, deterministic (compatible with TEN3 model, use to initialize)
class FwdCNN3(nn.Module):
    def __init__(self, opt, mfile):
        super(FwdCNN3, self).__init__()
        self.opt = opt
        # If we are given a model file, use it to initialize this model. 
        # otherwise initialize from scratch
        if mfile == '':
            self.encoder = encoder(opt, 0, opt.ncond)
            self.decoder = decoder(opt)
        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)['model']
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.a_encoder = pretrained_model.a_encoder
            self.u_network = pretrained_model.u_network
            self.encoder.n_inputs = opt.ncond


        self.a_encoder = nn.Sequential(
            nn.Linear(self.opt.n_actions, self.opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.opt.hidden_size)
        )

        self.u_network = u_network(opt)


    def forward(self, inputs, actions, target, sampling=None, p_dropout=None):
        npred = actions.size(1)
        input_images, input_states = inputs
        pred_images, pred_states, pred_costs = [], [], []
        for t in range(npred):
            h = self.encoder(input_images, input_states)
            a_emb = self.a_encoder(actions[:, t]).view(h.size())
            h = utils.combine(h, a_emb, self.opt.combine)
            h = h + self.u_network(h)

            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        return [pred_images, pred_states, pred_costs, None], Variable(torch.zeros(1).cuda())

    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()



# forward model, deterministic (compatible with TEN3 model, use to initialize)
class FwdCNN3_STN(nn.Module):
    def __init__(self, opt, mfile):
        super(FwdCNN3_STN, self).__init__()
        self.opt = opt
        # If we are given a model file, use it to initialize this model. 
        # otherwise initialize from scratch
        if mfile == '':
            self.encoder = encoder(opt, 0, opt.ncond)
            self.decoder = decoder(opt)
        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)['model']
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.a_encoder = pretrained_model.a_encoder
            self.u_network = pretrained_model.u_network
            self.encoder.n_inputs = opt.ncond


        self.a_encoder = nn.Sequential(
            nn.Linear(self.opt.n_actions, self.opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.opt.hidden_size)
        )

        self.u_network = u_network(opt)


    def forward(self, inputs, actions, target, sampling=None, p_dropout=None):
        npred = actions.size(1)
        input_images, input_states = inputs
        pred_images, pred_states, pred_costs = [], [], []
        for t in range(npred):
            h = self.encoder(input_images, input_states)
            a_emb = self.a_encoder(actions[:, t]).view(h.size())
            h = utils.combine(h, a_emb, self.opt.combine)
            h = h + self.u_network(h)

            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        return [pred_images, pred_states, pred_costs, None], Variable(torch.zeros(1).cuda())

    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()





class FwdCNN2(nn.Module):
    def __init__(self, opt, mfile):
        super(FwdCNN2, self).__init__()
        self.opt = opt
        # If we are given a model file, use it to initialize this model. 
        # otherwise initialize from scratch
        if mfile == '':
            self.encoder = encoder(opt, 0, opt.ncond)
            self.decoder = decoder(opt)
            self.a_network = nn.Sequential(
                nn.Linear(self.opt.n_actions, self.opt.nfeature), 
                nn.LeakyReLU(0.2, inplace=True), 
                nn.Linear(self.opt.nfeature, self.opt.nfeature), 
                nn.LeakyReLU(0.2, inplace=True), 
                nn.Linear(self.opt.nfeature, self.opt.hidden_size)
            )

            self.h_network = nn.Sequential(
                nn.Linear(self.opt.hidden_size, self.opt.n_hidden), 
                nn.LeakyReLU(0.2, inplace=True), 
                nn.Linear(self.opt.n_hidden, self.opt.n_hidden), 
                nn.LeakyReLU(0.2, inplace=True), 
                nn.Linear(self.opt.n_hidden, self.opt.n_hidden), 
                nn.LeakyReLU(0.2, inplace=True), 
                nn.Linear(self.opt.n_hidden, self.opt.hidden_size)
            )

        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.encoder.n_inputs = opt.ncond

    def forward(self, inputs, actions, target, sampling=None, p_dropout=None):
        t0 = time.time()
        bsize = actions.size(0)
        npred = actions.size(1)
        input_images, input_states = inputs
        pred_images, pred_states, pred_costs = [], [], []
        h = self.encoder(input_images, input_states)
        h = h.view(bsize, self.opt.hidden_size)
        for t in range(npred):
            h = h + self.a_network(actions[:, t])
            h = h + self.h_network(h)
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + pred_images_images[:, -1].unsqueeze(1))
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        print(time.time() - t0)
        return [pred_images, pred_states, pred_costs], Variable(torch.zeros(1).cuda())

    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()









# forward LSTM model, deterministic
class FwdLSTM(nn.Module):
    def __init__(self, opt, mfile):
        super(FwdCNN, self).__init__()
        self.opt = opt
        # If we are given a model file, use it to initialize this model. 
        # otherwise initialize from scratch
        if mfile == '':
            self.encoder = encoder(opt, opt.n_actions, 1)
            self.decoder = decoder(opt)
        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.encoder.n_inputs = opt.ncond

    def forward(self, inputs, actions, target, sampling=None):
        npred = actions.size(1)
        input_images, input_states = inputs
        pred_images, pred_states, pred_costs = [], [], []
        for t in range(npred):
            h = self.encoder(input_images, input_states, actions[:, t])
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        return [pred_images, pred_states, pred_costs], Variable(torch.zeros(1).cuda())

    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()
















# forward model, Mixture Density Network. (tried this for comparison, but it's really hard to train). 
class FwdCNN_MDN(nn.Module):
    def __init__(self, opt, mfile):
        super(FwdCNN_MDN, self).__init__()
        self.opt = opt
        self.encoder = encoder(opt, opt.n_actions, opt.ncond)
        self.decoder = decoder(opt, n_out = 2*opt.n_mixture)

        self.pi_network = nn.Sequential(
            nn.Linear(self.opt.hidden_size, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.n_mixture)
        )

    def forward(self, inputs, actions, targets, sampling=None):
        npred = actions.size(1)
        input_images, input_states = inputs
        bsize = actions.size(0)
        pred_images_mu, pred_states_mu, pred_costs_mu = [], [], []
        pred_images_sigma, pred_states_sigma, pred_costs_sigma = [], [], []
        latent_probs = []
        for t in range(npred):
            h = self.encoder(input_images, input_states, actions[:, t])
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = pred_image.view(bsize, self.opt.n_mixture, 2, 3, self.opt.height, self.opt.width)
            pred_image_mu = pred_image[:, :, 0].contiguous().view(bsize, -1)
            pred_image_sigma = pred_image[:, :, 1].contiguous().view(bsize, -1)
            pred_state = pred_state.view(bsize, self.opt.n_mixture, 2, 4)
            pred_state_mu = pred_state[:, :, 0].contiguous().view(bsize, -1)
            pred_state_sigma = pred_state[:, :, 1].contiguous().view(bsize, -1)
            pred_cost = pred_cost.view(bsize, self.opt.n_mixture, 2, 2)
            pred_cost_mu = pred_cost[:, :, 0].contiguous().view(bsize, -1)
            pred_cost_sigma = pred_cost[:, :, 1].contiguous().view(bsize, -1)

            pi = F.softmax(self.pi_network(h.view(bsize, -1)), dim=1)
            pred_image_mu = F.sigmoid(pred_image_mu)
            pred_image_sigma = F.softplus(pred_image_sigma)
            pred_state_sigma = F.softplus(pred_state_sigma)
            pred_cost_sigma = F.softplus(pred_cost_sigma)

            if targets is not None:
                target_images, target_states, target_costs = targets
                input_images = torch.cat((input_images[:, 1:], target_images[:, t].unsqueeze(1)), 1)
                input_states = torch.cat((input_states[:, 1:], target_states[:, t].unsqueeze(1)), 1)
            else:
                # sample, TODO
                pdb.set_trace()

            pred_images_mu.append(pred_image_mu)
            pred_states_mu.append(pred_state_mu)
            pred_costs_mu.append(pred_cost_mu)
            pred_images_sigma.append(pred_image_sigma)
            pred_states_sigma.append(pred_state_sigma)
            pred_costs_sigma.append(pred_cost_sigma)
            latent_probs.append(pi)


        pred_images_mu = torch.stack(pred_images_mu, 1)
        pred_states_mu = torch.stack(pred_states_mu, 1)
        pred_costs_mu = torch.stack(pred_costs_mu, 1)
        pred_images_sigma = torch.stack(pred_images_sigma, 1)
        pred_states_sigma = torch.stack(pred_states_sigma, 1)
        pred_costs_sigma = torch.stack(pred_costs_sigma, 1)
        latent_probs = torch.stack(latent_probs, 1)

        pred_images_sigma = torch.clamp(pred_images_sigma, min=1e-1)
        pred_states_sigma = torch.clamp(pred_states_sigma, min=1e-5)
        pred_costs_sigma = torch.clamp(pred_costs_sigma, min=1e-5)

        return [[pred_images_mu, pred_images_sigma], [pred_states_mu, pred_states_sigma], [pred_costs_mu, pred_costs_sigma], latent_probs], Variable(torch.zeros(1).cuda())


    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()






# forward TEN model (autoregressive)
class FwdCNN_TEN(nn.Module):
    def __init__(self, opt, mfile=''):
        super(FwdCNN_TEN, self).__init__()
        self.opt = opt

        if mfile == '':
            self.encoder = encoder(opt, opt.n_actions, opt.ncond)
            self.decoder = decoder(opt)
        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.encoder.n_inputs = opt.ncond
            self.decoder.n_out = 1

        self.y_encoder = encoder(opt, 0, 1, states=False)

        self.z_network = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nz)
        )

        self.z_zero = Variable(torch.zeros(self.opt.batch_size, self.opt.nz))


        # if beta > 0, it means we are training a prior network jointly
        if self.opt.beta > 0:
            self.prior_network = v_network(opt)

        self.z_expander = nn.Linear(opt.nz, opt.hidden_size)
        # this will store the non-parametric estimate of p(z)
        self.p_z = []

    # save a z vector in the memory buffer
    def save_z(self, z):
        if len(self.p_z) == 0:
            self.p_z = z.data.cpu()
        else:
            self.p_z = torch.cat((self.p_z, z.data.cpu()), 0)

    # compute the nearest-neighbor graph among z vectors
    def compute_z_graph(self):
        m = self.p_z.size(0)
        # do this in CUDA...
        self.knn_indx = torch.zeros(m, 4000).cuda() # indices
        self.knn_dist = torch.zeros(m, 4000).cuda() # distances
        self.p_z = self.p_z.cuda()
        for i in range(m):
            print('[computing z graph {}]'.format(float(i)/m), end="\r")
            d = torch.norm(self.p_z - self.p_z[i].view(1, self.opt.nz), 2, 1)
            nb_dist, nb_indx = torch.topk(d, 4000, largest=False)
            self.knn_indx[i].copy_(nb_indx)
            self.knn_dist[i].copy_(nb_dist)
        self.knn_dist = self.knn_dist.cpu()
        self.knn_indx = self.knn_indx.cpu().long()
            
    # sample a latent z vector, using uniform, KNN or learned prior methods. 
    def sample_z(self, bsize, method='fp', input_images=None, input_states=None, action=None, z_prev=None, t0=False):
        z_indx = None
        if method != 'knn':
            z = []
        # fixed prior, i.e. uniform
        if method == 'fp': 
            for b in range(bsize):
                z.append(random.choice(self.p_z))
            z = torch.stack(z).contiguous()
        # at each timestep, sample on of the K nearest neighbors of the current z vector
        elif method == 'knn':
            if t0:
                # we only have the initial z vector, so quantize it first to one in the training set
                self.p_z = self.p_z.cuda()
                d = torch.norm(self.p_z.view(-1, 1, self.opt.nz) - z_prev.view(1, -1, self.opt.nz).data.cuda(), 2, 2)
                z_prev_indx = torch.topk(d, 1, dim=0, largest=False)[1].squeeze()
            else:
                z_prev, z_prev_indx = z_prev
            z, z_indx = [], []
            for b in range(bsize):
                indx = random.choice(self.knn_indx[z_prev_indx[b]][:self.opt.topz_sample])
                z_indx.append(indx)
                z.append(self.p_z[indx])
            z = torch.stack(z).contiguous()            
        # use a learned likelihood model
        elif method == 'pdf':
            M = self.p_z.size(0)
            nz = self.p_z.size(1)
            pi, mu, sigma = self.prior_network(input_images, input_states, action)
            bsize = mu.size(0)
            # sample a z vector from the likelihood model
            pi_sample = torch.multinomial(pi, 1).squeeze()
            mu_sample = torch.gather(mu, dim=1, index=pi_sample.view(bsize, 1, 1).expand(bsize, 1, nz)).squeeze()
            sigma_sample = torch.gather(sigma, dim=1, index=pi_sample.view(bsize, 1, 1).expand(bsize, 1, nz)).squeeze()
            z_sample = mu_sample.data + torch.randn(bsize, nz).cuda()*sigma_sample.data
            # quantize it to its nearest neighbor
            dist = torch.norm(self.p_z.view(-1, 1, nz) - z_sample.view(1, -1, nz), 2, 2)
            _, z_ind = torch.topk(dist, 1, dim=0, largest=False)
            z = self.p_z.index(z_ind.squeeze().cuda())
        elif method == 'hinge_knn':
            M = self.p_z.size(0)
            nz = self.p_z.size(1)
            h_x = self.encoder(input_images, input_states, action)
            u = self.prior_network(h_x).data
            bsize = u.size(0)
            sim = torch.sum(self.p_z.view(-1, 1, nz)*u.view(1, -1, nz), 2)
            topsim, z_ind = torch.topk(sim, 100, dim=0, largest=True)
            pdb.set_trace()
            z = []
            for b in range(bsize):
                indx = random.choice(z_ind[:, b])
                z.append(self.p_z[indx])
            z = torch.stack(z).contiguous()
            
            
        if self.use_cuda: z = z.cuda()
        return [Variable(z), z_indx]

    def forward(self, inputs, actions, targets, save_z = False, sampling=None, p_dropout=0.0, z_seq=None):
        input_images, input_states = inputs
        bsize = input_images.size(0)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)
        ploss = Variable(torch.zeros(1))
        if self.use_cuda:
            ploss = ploss.cuda()

        pred_images, pred_states, pred_costs = [], [], []
        self.Z = []
        self.z_top_list = []
            
        z = None
        for t in range(npred):
            # encode the inputs
            h_x = self.encoder(input_images, input_states, actions[:, t])
            if sampling is None:
                # we are training or estimating z distribution
                target_images, target_states, target_costs = targets
                # encode the targets into z
                h_y = self.y_encoder(target_images[:, t].unsqueeze(1).contiguous())
                if random.random() < p_dropout:
                    z = self.z_zero
                else:
                    z = self.z_network(utils.combine(h_x, h_y, self.opt.combine).view(bsize, -1))
                    if self.opt.z_sphere == 1:
                        z = z / torch.norm(z, 2, 1).unsqueeze(1)
                    if save_z:
                        self.save_z(z)
                    if self.opt.beta > 0:
                        u = self.prior_network(h_x)
                        ploss = utils.hinge_loss(u, z)
                        '''
                        pi, mu, sigma = self.prior_network(h_x)
                        ploss = utils.mdn_loss_fn(pi, sigma, mu, z)
                        '''
                        if math.isnan(ploss.data[0]):
                            pdb.set_trace()
            else:
                # we are doing inference
                if z_seq is not None:
                    z = [z_seq[t], None]
                else:
                    z = self.sample_z(bsize, sampling, input_images, input_states, actions[:, t], z, t0=False)

            z_ = z if sampling is None else z[0]
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp, self.opt.combine)

            pred_image, pred_state, pred_cost = self.decoder(h)
            if sampling is not None:
                pred_image.detach()
                pred_state.detach()
                pred_cost.detach()
            if self.opt.sigmoid_out == 1:
                pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            else:
                pred_image = torch.clamp(pred_image + input_images[:, -1].unsqueeze(1), min=0, max=1)
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        return [pred_images, pred_states, pred_costs], ploss


    def plan_actions_backprop(self, observation, args, verbose=True, normalize=True, optimize_z=False, optimize_a=True, gamma=0.99):
        input_images, input_states = observation
        input_images = input_images.float().clone()
        input_states = input_states.clone()
        if normalize:
            input_images.div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = Variable(input_images.cuda()).unsqueeze(0)
            input_states = Variable(input_states.cuda()).unsqueeze(0)
        bsize = input_images.size(0)
        # repeat for multiple rollouts
        input_images = input_images.unsqueeze(1).expand(bsize, args.n_rollouts, args.ncond, 3, args.height, args.width)
        input_states = input_states.unsqueeze(1).expand(bsize, args.n_rollouts, args.ncond, 4)
        input_images = input_images.contiguous().view(bsize * args.n_rollouts, args.ncond, 3, args.height, args.width)
        input_states = input_states.contiguous().view(bsize * args.n_rollouts, args.ncond, 4)
        actions = Variable(torch.randn(bsize, args.npred, self.opt.n_actions).cuda().mul_(0.0))
        if optimize_a:
            actions.requires_grad = True
            optimizer_a = optim.Adam([actions], args.lrt)

        Z = self.sample_z(bsize * args.n_rollouts * args.npred, method='fp')[0]
        Z = Z.view(args.npred, bsize * args.n_rollouts, -1)
        Z0 = Z.clone()
        gamma_mask = Variable(torch.from_numpy(numpy.array([[gamma**t, gamma**t] for t in range(args.npred)])).float().cuda()).unsqueeze(0)
        if optimize_z:
            Z.requires_grad = True
            optimizer_z = optim.Adam([Z], args.lrt)
        pred = None
        for i in range(0, args.n_iter):
            optimizer_a.zero_grad()
            self.zero_grad()
            actions_rep = actions.unsqueeze(1).expand(bsize, args.n_rollouts, args.npred, 2).contiguous().view(bsize * args.n_rollouts, args.npred, 2)
            pred, _ = self.forward([input_images, input_states], actions_rep, None, sampling='fp', z_seq=Z)
            costs = pred[2]
            weighted_costs = costs * gamma_mask
            loss = weighted_costs[:, :, 0].mean() + 0.5*weighted_costs[:, :, 1].mean()
            loss.backward()
            if verbose:
                print('[iter {} | mean pred cost = ({:.4f}, {:.4f})], grad = {}'.format(i, costs.data[:, :, 0].mean(), costs.data[:, :, 1].mean(), actions.grad.data.norm()))
            if optimize_a:
                torch.nn.utils.clip_grad_norm([actions], 1)
                optimizer_a.step()
                actions.data.clamp_(min=-2, max=2)
                actions.data[:, :, 1].clamp_(min=-1, max=1)

            if optimize_z:
                torch.nn.utils.clip_grad_norm([Z], 1)
                Z.grad *= -1
                optimizer_z.step()


        # evaluate on new futures
        Z_test = self.sample_z(bsize * args.n_rollouts * args.npred, method='fp')[0]
        Z_test = Z_test.view(args.npred, bsize * args.n_rollouts, -1)
        actions_rep = actions.unsqueeze(1).expand(bsize, args.n_rollouts, args.npred, 2).contiguous().view(bsize * args.n_rollouts, args.npred, 2)
        pred, _ = self.forward([input_images, input_states], actions_rep, None, sampling='fp', z_seq=Z_test)
        costs = pred[2]
        loss_test = costs[:, :, 0].mean() + 0.5*costs[:, :, 1].mean()
        print('\n[pred test cost = {}]\n'.format(loss_test.data[0]))
        
        '''
        # also get predictions using zero actions
        pred_const, _ = self.forward([input_images, input_states], actions.clone().zero_(), None, sampling='fp', z_seq=Z0)
        '''
        pred_const = None
        actions = actions.data.cpu()
        if normalize:
            actions *= self.stats['a_std'].view(1, 1, 2).expand(actions.size())
            actions += self.stats['a_mean'].view(1, 1, 2).expand(actions.size())
        return actions.squeeze().numpy(), pred, pred_const, loss_test.data[0]


    def create_policy_net(self, opt):
        if opt.targetprop == 0:
            self.policy_net = StochasticPolicy(opt)
        elif opt.targetprop == 1:
            self.policy_net = PolicyMDN(opt)


    def train_policy_net(self, inputs, targets, targetprop=0):
        t0 = time.time()
        input_images, input_states = inputs
        target_images, target_states, target_costs = targets
        bsize = input_images.size(0)
        npred = target_images.size(1)
        pred_images, pred_states, pred_costs, pred_actions = [], [], [], []
            
        z = None
        for t in range(npred):
            # encode the inputs
            if targetprop == 0:
                actions = self.policy_net(input_images, input_states)
            elif targetprop == 1:
                _, _, _, actions = self.policy_net(input_images, input_states, sample=True)
                actions = Variable(actions)
            h_x = self.encoder(input_images, input_states, actions)
            target_images, target_states, target_costs = targets
            # encode the targets into z
            h_y = self.y_encoder(torch.cat((input_images, target_images[:, t].unsqueeze(1).contiguous()), 1), input_states, actions)
            z = self.z_network(utils.combine(h_x, h_y, self.opt.combine).view(bsize, -1))
            z_ = z
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)
            pred_actions.append(actions)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        pred_actions = torch.stack(pred_actions, 1)
#        print(time.time() - t0)
        return [pred_images, pred_states, pred_costs], pred_actions



    # assuming this is for a single input sample
    def compute_action_policy_net(self, inputs, opt, npred=50, n_actions=10, n_futures=5, normalize=True):
        input_images, input_states = inputs
        pred_images, pred_states, pred_costs = [], [], []

        if normalize:
            input_images = input_images.clone().float().div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = Variable(input_images.cuda()).unsqueeze(0)
            input_states = Variable(input_states.cuda()).unsqueeze(0)


        # repeat for multiple rollouts
        bsize = n_futures * n_actions
        input_images = input_images.expand(bsize, opt.ncond, 3, opt.height, opt.width)
        input_states = input_states.expand(bsize, opt.ncond, 4)
        input_images = input_images.contiguous().view(bsize, opt.ncond, 3, opt.height, opt.width)
        input_states = input_states.contiguous().view(bsize, opt.ncond, 4)
            
        Z = self.sample_z(n_futures * npred, method='fp')[0]
        Z = Z.view(npred, 1, n_futures, -1)
        Z = Z.expand(npred, n_actions, n_futures, -1)
        Z0 = Z.clone()

        # sample initial actions: we will choose to execute one of these at the end
        actions_init = self.policy_net(input_images[0].unsqueeze(0), input_states[0].unsqueeze(0), sample=True, normalize=False, n_samples=n_actions)
        actions = actions.clone().unsqueeze(0).expand(n_futures, n_actions, 2)
        for t in range(npred):
            # encode the inputs
            actions = self.policy_net(input_images, input_states)
            h_x = self.encoder(input_images, input_states, actions)
            z_ = Z[t]
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        return [pred_images, pred_states, pred_costs], None


        
    def intype(self, t):
        if t == 'gpu':
            self.cuda()
            self.z_zero = self.z_zero.cuda()
            self.use_cuda = True
            if len(self.p_z) > 0:
                self.p_z = self.p_z.cuda()
        elif t == 'cpu':
            self.cpu()
            self.use_cuda = False
            self.z_zero = self.z_zero.cpu()
            if len(self.p_z) > 0:
                self.p_z = self.p_z.cpu()














































# forward TEN model (autoregressive)
class FwdCNN_TEN2(nn.Module):
    def __init__(self, opt, mfile=''):
        super(FwdCNN_TEN2, self).__init__()
        self.opt = opt

        if mfile == '':
            self.encoder = encoder(opt, 0, opt.ncond)
            self.decoder = decoder(opt)


            self.a_network = nn.Sequential(
                nn.Linear(self.opt.n_actions, self.opt.nfeature), 
                nn.PReLU(self.opt.nfeature, init=1), 
                nn.Linear(self.opt.nfeature, self.opt.nfeature), 
                nn.PReLU(self.opt.nfeature, init=1), 
                nn.Linear(self.opt.nfeature, self.opt.hidden_size)
            )


            self.h_network_conv1 = nn.Sequential(
                nn.Conv2d(self.opt.nfeature, self.opt.nfeature, 1, 1), 
                nn.PReLU(self.opt.nfeature, init=1),
                nn.Conv2d(self.opt.nfeature, self.opt.nfeature, 1, 1)
            )


            self.h_network1 = nn.Sequential(
                nn.Linear(self.opt.hidden_size, self.opt.n_hidden), 
                nn.PReLU(self.opt.n_hidden, init=1), 
                nn.Linear(self.opt.n_hidden, self.opt.n_hidden), 
                nn.PReLU(self.opt.n_hidden, init=1), 
                nn.Linear(self.opt.n_hidden, self.opt.hidden_size)
            )


            self.h_network_conv2 = nn.Sequential(
                nn.Conv2d(self.opt.nfeature, self.opt.nfeature, 1, 1), 
                nn.PReLU(self.opt.nfeature, init=1),
                nn.Conv2d(self.opt.nfeature, self.opt.nfeature, 1, 1)
            )

            self.h_network2 = nn.Sequential(
                nn.Linear(self.opt.hidden_size, self.opt.n_hidden), 
                nn.PReLU(self.opt.n_hidden, init=1), 
                nn.Linear(self.opt.n_hidden, self.opt.n_hidden), 
                nn.PReLU(self.opt.n_hidden, init=1), 
                nn.Linear(self.opt.n_hidden, self.opt.hidden_size)
            )


        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.encoder.n_inputs = opt.ncond
            self.decoder.n_out = 1

        self.y_encoder = encoder(opt, 0, 1)

        self.z_network = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.nfeature),
            nn.PReLU(self.opt.nfeature, init=1),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.PReLU(self.opt.nfeature, init=1),
            nn.Linear(opt.nfeature, opt.nz)
        )

        # if beta > 0, it means we are training a prior network jointly
        if self.opt.beta > 0:
            self.prior_network = v_network(opt)

        self.z_expander = nn.Linear(opt.nz, opt.hidden_size)
        # this will store the non-parametric estimate of p(z)
        self.p_z = []

    # save a z vector in the memory buffer
    def save_z(self, z):
        if len(self.p_z) == 0:
            self.p_z = z.data.cpu()
        else:
            self.p_z = torch.cat((self.p_z, z.data.cpu()), 0)

            
    # sample a latent z vector, using uniform, KNN or learned prior methods. 
    def sample_z(self, bsize, method='fp', input_images=None, input_states=None, action=None, z_prev=None, t0=False):
        z_indx = None
        if method != 'knn':
            z = []
        # fixed prior, i.e. uniform
        if method == 'fp': 
            for b in range(bsize):
                z.append(random.choice(self.p_z))
            z = torch.stack(z).contiguous()
        # at each timestep, sample on of the K nearest neighbors of the current z vector
        elif method == 'knn':
            if t0:
                # we only have the initial z vector, so quantize it first to one in the training set
                self.p_z = self.p_z.cuda()
                d = torch.norm(self.p_z.view(-1, 1, self.opt.nz) - z_prev.view(1, -1, self.opt.nz).data.cuda(), 2, 2)
                z_prev_indx = torch.topk(d, 1, dim=0, largest=False)[1].squeeze()
            else:
                z_prev, z_prev_indx = z_prev
            z, z_indx = [], []
            for b in range(bsize):
                indx = random.choice(self.knn_indx[z_prev_indx[b]][:self.opt.topz_sample])
                z_indx.append(indx)
                z.append(self.p_z[indx])
            z = torch.stack(z).contiguous()            
        # use a learned likelihood model
        elif method == 'pdf':
            M = self.p_z.size(0)
            nz = self.p_z.size(1)
            pi, mu, sigma = self.prior_network(input_images, input_states, action)
            bsize = mu.size(0)
            # sample a z vector from the likelihood model
            pi_sample = torch.multinomial(pi, 1).squeeze()
            mu_sample = torch.gather(mu, dim=1, index=pi_sample.view(bsize, 1, 1).expand(bsize, 1, nz)).squeeze()
            sigma_sample = torch.gather(sigma, dim=1, index=pi_sample.view(bsize, 1, 1).expand(bsize, 1, nz)).squeeze()
            z_sample = mu_sample.data + torch.randn(bsize, nz).cuda()*sigma_sample.data
            # quantize it to its nearest neighbor
            dist = torch.norm(self.p_z.view(-1, 1, nz) - z_sample.view(1, -1, nz), 2, 2)
            _, z_ind = torch.topk(dist, 1, dim=0, largest=False)
            z = self.p_z.index(z_ind.squeeze().cuda())
        elif method == 'hinge_knn':
            M = self.p_z.size(0)
            nz = self.p_z.size(1)
            h_x = self.encoder(input_images, input_states, action)
            u = self.prior_network(h_x).data
            bsize = u.size(0)
            sim = torch.sum(self.p_z.view(-1, 1, nz)*u.view(1, -1, nz), 2)
            topsim, z_ind = torch.topk(sim, 100, dim=0, largest=True)
            pdb.set_trace()
            z = []
            for b in range(bsize):
                indx = random.choice(z_ind[:, b])
                z.append(self.p_z[indx])
            z = torch.stack(z).contiguous()
            
            
        if self.use_cuda: z = z.cuda()
        return [Variable(z), z_indx]

    def forward(self, inputs, actions, targets, save_z = False, sampling=None, p_dropout=0.0, z_seq=None):
        input_images, input_states = inputs
        bsize = input_images.size(0)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)
        ploss = Variable(torch.zeros(1))
        if self.use_cuda:
            ploss = ploss.cuda()

        pred_images, pred_states, pred_costs = [], [], []
        self.Z = []
        self.z_top_list = []
            
        z = None
        # encode the inputs
        h = self.encoder(input_images, input_states).view(bsize, self.opt.hidden_size)

        for t in range(npred):
            h = h + self.a_network(actions[:, t]).view(bsize, self.opt.hidden_size)
            h = h.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = self.h_network_conv1(h)
            h = h.view(bsize, self.opt.hidden_size)
            h = h + self.h_network1(h)
            if sampling is None:
                # we are training or estimating z distribution
                target_images, target_states, target_costs = targets
                # encode the targets into z
                h_y = self.y_encoder(target_images[:, t].unsqueeze(1).contiguous()).view(bsize, self.opt.hidden_size)
                if random.random() < p_dropout:
                    z = Variable(torch.zeros(bsize, self.opt.nz).cuda())
                else:
                    z = self.z_network(utils.combine(h, h_y, self.opt.combine).view(bsize, -1))
                    if self.opt.z_sphere == 1:
                        z = z / torch.norm(z, 2, 1).unsqueeze(1)
                    if save_z:
                        self.save_z(z)
                    if self.opt.beta > 0:
                        u = self.prior_network(h_x)
                        ploss = utils.hinge_loss(u, z)
                        '''
                        pi, mu, sigma = self.prior_network(h_x)
                        ploss = utils.mdn_loss_fn(pi, sigma, mu, z)
                        '''
                        if math.isnan(ploss.data[0]):
                            pdb.set_trace()
            else:
                # we are doing inference
                if z_seq is not None:
                    z = [z_seq[t], None]
                else:
                    z = self.sample_z(bsize, sampling, input_images, input_states, actions[:, t], z, t0=False)

            z_ = z if sampling is None else z[0]
            z_exp = self.z_expander(z_).view(bsize, self.opt.hidden_size)
            h = utils.combine(h, z_exp, self.opt.combine)
            h = h.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = self.h_network_conv2(h)
            h = h.view(bsize, self.opt.hidden_size)
            h = h + self.h_network2(h)
            pred_image, pred_state, pred_cost = self.decoder(h.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width))
            if sampling is not None:
                pred_image.detach()
                pred_state.detach()
                pred_cost.detach()
            if t == 0:
                last_image = input_images[:, -1].unsqueeze(1)
                last_state = input_states[:, -1]
            else:
                last_image = pred_images[-1]
                last_state = pred_states[-1]
                
            pred_image = torch.clamp(pred_image + last_image, min=0, max=1)
            pred_state = torch.clamp(pred_state + last_state, min=-6, max=6)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        return [pred_images, pred_states, pred_costs], ploss


    def plan_actions_backprop(self, observation, args, verbose=True, normalize=True, optimize_z=False, optimize_a=True, gamma=0.99):
        input_images, input_states = observation
        input_images = input_images.float().clone()
        input_states = input_states.clone()
        if normalize:
            input_images.div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = Variable(input_images.cuda()).unsqueeze(0)
            input_states = Variable(input_states.cuda()).unsqueeze(0)
        bsize = input_images.size(0)
        # repeat for multiple rollouts
        input_images = input_images.unsqueeze(1).expand(bsize, args.n_rollouts, args.ncond, 3, args.height, args.width)
        input_states = input_states.unsqueeze(1).expand(bsize, args.n_rollouts, args.ncond, 4)
        input_images = input_images.contiguous().view(bsize * args.n_rollouts, args.ncond, 3, args.height, args.width)
        input_states = input_states.contiguous().view(bsize * args.n_rollouts, args.ncond, 4)
        actions = Variable(torch.randn(bsize, args.npred, self.opt.n_actions).cuda().mul_(0.0))
        if optimize_a:
            actions.requires_grad = True
            optimizer_a = optim.Adam([actions], args.lrt)

        Z = self.sample_z(bsize * args.n_rollouts * args.npred, method='fp')[0]
        Z = Z.view(args.npred, bsize * args.n_rollouts, -1)
        Z0 = Z.clone()
        gamma_mask = Variable(torch.from_numpy(numpy.array([[gamma**t, gamma**t] for t in range(args.npred)])).float().cuda()).unsqueeze(0)
        if optimize_z:
            Z.requires_grad = True
            optimizer_z = optim.Adam([Z], args.lrt)
        pred = None
        for i in range(0, args.n_iter):
            optimizer_a.zero_grad()
            self.zero_grad()
            actions_rep = actions.unsqueeze(1).expand(bsize, args.n_rollouts, args.npred, 2).contiguous().view(bsize * args.n_rollouts, args.npred, 2)
            pred, _ = self.forward([input_images, input_states], actions_rep, None, sampling='fp', z_seq=Z)
            costs = pred[2]
            weighted_costs = costs * gamma_mask
            loss = weighted_costs[:, :, 0].mean() + 0.5*weighted_costs[:, :, 1].mean()
            loss.backward()
            if verbose:
                print('[iter {} | mean pred cost = ({:.4f}, {:.4f})], grad = {}'.format(i, costs.data[:, :, 0].mean(), costs.data[:, :, 1].mean(), actions.grad.data.norm()))
            if optimize_a:
                torch.nn.utils.clip_grad_norm([actions], 1)
                optimizer_a.step()
                actions.data.clamp_(min=-2, max=2)
                actions.data[:, :, 1].clamp_(min=-1, max=1)

            if optimize_z:
                torch.nn.utils.clip_grad_norm([Z], 1)
                Z.grad *= -1
                optimizer_z.step()


        # evaluate on new futures
        Z_test = self.sample_z(bsize * args.n_rollouts * args.npred, method='fp')[0]
        Z_test = Z_test.view(args.npred, bsize * args.n_rollouts, -1)
        actions_rep = actions.unsqueeze(1).expand(bsize, args.n_rollouts, args.npred, 2).contiguous().view(bsize * args.n_rollouts, args.npred, 2)
        pred, _ = self.forward([input_images, input_states], actions_rep, None, sampling='fp', z_seq=Z_test)
        costs = pred[2]
        loss_test = costs[:, :, 0].mean() + 0.5*costs[:, :, 1].mean()
        print('\n[pred test cost = {}]\n'.format(loss_test.data[0]))
        
        '''
        # also get predictions using zero actions
        pred_const, _ = self.forward([input_images, input_states], actions.clone().zero_(), None, sampling='fp', z_seq=Z0)
        '''
        pred_const = None
        actions = actions.data.cpu()
        if normalize:
            actions *= self.stats['a_std'].view(1, 1, 2).expand(actions.size())
            actions += self.stats['a_mean'].view(1, 1, 2).expand(actions.size())
        return actions.squeeze().numpy(), pred, pred_const, loss_test.data[0]


    def create_policy_net(self, opt):
        if opt.targetprop == 0:
            self.policy_net = StochasticPolicy(opt)
        elif opt.targetprop == 1:
            self.policy_net = PolicyMDN(opt)


    def train_policy_net(self, inputs, targets, targetprop=0):
        t0 = time.time()
        input_images, input_states = inputs
        target_images, target_states, target_costs = targets
        bsize = input_images.size(0)
        npred = target_images.size(1)
        pred_images, pred_states, pred_costs, pred_actions = [], [], [], []
            
        z = None
        for t in range(npred):
            # encode the inputs
            if targetprop == 0:
                actions = self.policy_net(input_images, input_states)
            elif targetprop == 1:
                _, _, _, actions = self.policy_net(input_images, input_states, sample=True)
                actions = Variable(actions)
            h_x = self.encoder(input_images, input_states, actions)
            target_images, target_states, target_costs = targets
            # encode the targets into z
            h_y = self.y_encoder(torch.cat((input_images, target_images[:, t].unsqueeze(1).contiguous()), 1), input_states, actions)
            z = self.z_network(utils.combine(h_x, h_y, self.opt.combine).view(bsize, -1))
            z_ = z
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)
            pred_actions.append(actions)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        pred_actions = torch.stack(pred_actions, 1)
#        print(time.time() - t0)
        return [pred_images, pred_states, pred_costs], pred_actions


    def train_policy_net_targetprop(self, inputs, targets, opt):
        input_images_, input_states_ = inputs
        target_images, target_states, target_costs = targets
        bsize = input_images_.size(0)
        npred = target_images.size(1)
        policy_input_images, policy_input_states = [], []

        gamma_mask = torch.Tensor([opt.gamma**t for t in range(opt.npred)]).view(1, -1).cuda()

            
        z = None
        loss_i, loss_s, loss_c = None, None, None
        self.actions = Variable(torch.randn(bsize, npred, self.opt.n_actions).cuda().mul_(0.1), requires_grad=True)
        self.optimizer_a = optim.Adam([self.actions], opt.lrt_traj)

        for i in range(opt.niter_traj):
            pred_images, pred_states, pred_costs = [], [], []
            policy_input_images, policy_input_states = [], []
            self.optimizer_a.zero_grad()
            self.zero_grad()
            for t in range(npred):
                # encode the inputs
                input_images = Variable(input_images_.data)
                input_states = Variable(input_states_.data)
                policy_input_images.append(input_images)
                policy_input_states.append(input_states)
                if i == 0:
                    _, _, _, a = self.policy_net(input_images, input_states, sample=True)
                    self.actions[:, t].data.copy_(a.squeeze())
                h_x = self.encoder(input_images, input_states, self.actions[:, t])
                target_images, target_states, target_costs = targets
                # encode the targets into z
                h_y = self.y_encoder(torch.cat((input_images, target_images[:, t].unsqueeze(1).contiguous()), 1), input_states, self.actions[:, t])
                z = self.z_network(utils.combine(h_x, h_y, self.opt.combine).view(bsize, -1))
                z_ = z
                z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
                h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
                h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)
                pred_image, pred_state, pred_cost = self.decoder(h)
                pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
                # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
                pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
                input_images = torch.cat((input_images[:, 1:], pred_image), 1)
                input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
                pred_images.append(pred_image)
                pred_states.append(pred_state)
                pred_costs.append(pred_cost)

            pred_images = torch.cat(pred_images, 1)
            pred_states = torch.stack(pred_states, 1)
            pred_costs = torch.stack(pred_costs, 1)
            loss_i = F.mse_loss(pred_images, target_images, reduce=False).mean(4).mean(3).mean(2)
            loss_s = F.mse_loss(pred_states, target_states, reduce=False).mean(2)

            proximity_cost = pred_costs[:, :, 0]
            lane_cost = pred_costs[:, :, 1]
            proximity_cost = proximity_cost * Variable(gamma_mask)
            lane_cost = lane_cost * Variable(gamma_mask)
            loss_c = proximity_cost.mean() + opt.lambda_lane * lane_cost.mean()
            loss_i = loss_i * Variable(gamma_mask)
            loss_s = loss_s * Variable(gamma_mask)
            loss_i = loss_i.mean()
            loss_s = loss_s.mean()
            loss = loss_i + loss_s + opt.lambda_c*loss_c
            loss.backward()
            self.optimizer_a.step()

        policy_input_images = torch.stack(policy_input_images).view(bsize*npred, self.opt.ncond, 3, self.opt.height, self.opt.width)
        policy_input_states = torch.stack(policy_input_states).view(bsize*npred, self.opt.ncond, 4)
        self.actions = self.actions.permute(1, 0, 2).contiguous().view(bsize*npred, 2)
        pi, mu, sigma, _ = self.policy_net(policy_input_images, policy_input_states)
        loss_mdn = utils.mdn_loss_fn(pi, sigma, mu, self.actions)
        self.optim_stats = self.optimizer_a.state_dict()
        return loss_i, loss_s, loss_c, loss_mdn



    # assuming this is for a single input sample
    def compute_action_policy_net(self, inputs, opt, npred=50, n_actions=10, n_futures=5, normalize=True):
        input_images, input_states = inputs
        pred_images, pred_states, pred_costs = [], [], []

        if normalize:
            input_images = input_images.clone().float().div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = Variable(input_images.cuda()).unsqueeze(0)
            input_states = Variable(input_states.cuda()).unsqueeze(0)


        # repeat for multiple rollouts
        bsize = n_futures * n_actions
        input_images = input_images.expand(bsize, opt.ncond, 3, opt.height, opt.width)
        input_states = input_states.expand(bsize, opt.ncond, 4)
        input_images = input_images.contiguous().view(bsize, opt.ncond, 3, opt.height, opt.width)
        input_states = input_states.contiguous().view(bsize, opt.ncond, 4)
            
        Z = self.sample_z(n_futures * npred, method='fp')[0]
        Z = Z.view(npred, 1, n_futures, -1)
        Z = Z.expand(npred, n_actions, n_futures, -1)
        Z0 = Z.clone()

        # sample initial actions: we will choose to execute one of these at the end
        actions_init = self.policy_net(input_images[0].unsqueeze(0), input_states[0].unsqueeze(0), sample=True, normalize=False, n_samples=n_actions)
        actions = actions.clone().unsqueeze(0).expand(n_futures, n_actions, 2)
        for t in range(npred):
            # encode the inputs
            actions = self.policy_net(input_images, input_states)
            h_x = self.encoder(input_images, input_states, actions)
            z_ = Z[t]
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        return [pred_images, pred_states, pred_costs], None


        
    def intype(self, t):
        if t == 'gpu':
            self.cuda()
            self.use_cuda = True
            if len(self.p_z) > 0:
                self.p_z = self.p_z.cuda()
        elif t == 'cpu':
            self.cpu()
            self.use_cuda = False
            if len(self.p_z) > 0:
                self.p_z = self.p_z.cpu()















# this version adds the actions *after* the z variables
class FwdCNN_TEN3(nn.Module):
    def __init__(self, opt, mfile=''):
        super(FwdCNN_TEN3, self).__init__()
        self.opt = opt

        if mfile == '':
            self.encoder = encoder(opt, 0, opt.ncond)
            self.decoder = decoder(opt)
        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)['model']
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.a_encoder = pretrained_model.a_encoder
            self.u_network = pretrained_model.u_network
            self.encoder.n_inputs = opt.ncond
            self.decoder.n_out = 1

        self.y_encoder = encoder(opt, 0, 1, states=False)

        self.a_encoder = nn.Sequential(
            nn.Linear(self.opt.n_actions, self.opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.opt.hidden_size)
        )

        self.u_network = u_network(opt)

        self.z_network = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nz)
        )

        self.z_zero = Variable(torch.zeros(self.opt.batch_size, self.opt.nz))

        if hasattr(self, 'action_indep_net'):
            if self.opt.action_indep_net == 1:
                self.action_indep_net = action_indep_net(opt)


        # if beta > 0, it means we are training a prior network jointly
        if self.opt.beta > 0:
            self.prior_network = v_network(opt)

        self.z_expander = nn.Linear(opt.nz, opt.hidden_size)
        # this will store the non-parametric estimate of p(z)
        self.p_z = []

    # save a z vector in the memory buffer
    def save_z(self, z):
        if len(self.p_z) == 0:
            self.p_z = z.data.cpu()
        else:
            self.p_z = torch.cat((self.p_z, z.data.cpu()), 0)

    # compute the nearest-neighbor graph among z vectors
    def compute_z_graph(self):
        m = self.p_z.size(0)
        # do this in CUDA...
        self.knn_indx = torch.zeros(m, 4000).cuda() # indices
        self.knn_dist = torch.zeros(m, 4000).cuda() # distances
        self.p_z = self.p_z.cuda()
        for i in range(m):
            print('[computing z graph {}]'.format(float(i)/m), end="\r")
            d = torch.norm(self.p_z - self.p_z[i].view(1, self.opt.nz), 2, 1)
            nb_dist, nb_indx = torch.topk(d, 4000, largest=False)
            self.knn_indx[i].copy_(nb_indx)
            self.knn_dist[i].copy_(nb_dist)
        self.knn_dist = self.knn_dist.cpu()
        self.knn_indx = self.knn_indx.cpu().long()
            
    # sample a latent z vector, using uniform, KNN or learned prior methods. 
    def sample_z(self, bsize, method='fp', input_images=None, input_states=None, action=None, z_prev=None, t0=False):
        z_indx = None
        if method != 'knn':
            z = []
        # fixed prior, i.e. uniform
        if method == 'fp': 
            for b in range(bsize):
                z.append(random.choice(self.p_z))
            z = torch.stack(z).contiguous()
        # at each timestep, sample on of the K nearest neighbors of the current z vector
        elif method == 'knn':
            if t0:
                # we only have the initial z vector, so quantize it first to one in the training set
                self.p_z = self.p_z.cuda()
                d = torch.norm(self.p_z.view(-1, 1, self.opt.nz) - z_prev.view(1, -1, self.opt.nz).data.cuda(), 2, 2)
                z_prev_indx = torch.topk(d, 1, dim=0, largest=False)[1].squeeze()
            else:
                z_prev, z_prev_indx = z_prev
            z, z_indx = [], []
            for b in range(bsize):
                indx = random.choice(self.knn_indx[z_prev_indx[b]][:self.opt.topz_sample])
                z_indx.append(indx)
                z.append(self.p_z[indx])
            z = torch.stack(z).contiguous()            
        # use a learned likelihood model
        elif method == 'pdf':
            M = self.p_z.size(0)
            nz = self.p_z.size(1)
            pi, mu, sigma = self.prior_network(input_images, input_states, action)
            bsize = mu.size(0)
            # sample a z vector from the likelihood model
            pi_sample = torch.multinomial(pi, 1).squeeze()
            mu_sample = torch.gather(mu, dim=1, index=pi_sample.view(bsize, 1, 1).expand(bsize, 1, nz)).squeeze()
            sigma_sample = torch.gather(sigma, dim=1, index=pi_sample.view(bsize, 1, 1).expand(bsize, 1, nz)).squeeze()
            z_sample = mu_sample.data + torch.randn(bsize, nz).cuda()*sigma_sample.data
            # quantize it to its nearest neighbor
            dist = torch.norm(self.p_z.view(-1, 1, nz) - z_sample.view(1, -1, nz), 2, 2)
            _, z_ind = torch.topk(dist, 1, dim=0, largest=False)
            z = self.p_z.index(z_ind.squeeze().cuda())
        elif method == 'hinge_knn':
            M = self.p_z.size(0)
            nz = self.p_z.size(1)
            h_x = self.encoder(input_images, input_states, action)
            u = self.prior_network(h_x).data
            bsize = u.size(0)
            sim = torch.sum(self.p_z.view(-1, 1, nz)*u.view(1, -1, nz), 2)
            topsim, z_ind = torch.topk(sim, 100, dim=0, largest=True)
            pdb.set_trace()
            z = []
            for b in range(bsize):
                indx = random.choice(z_ind[:, b])
                z.append(self.p_z[indx])
            z = torch.stack(z).contiguous()
            
            
        if self.use_cuda: z = z.cuda()
        return [Variable(z), z_indx]

    def forward(self, inputs, actions, targets, save_z = False, sampling=None, p_dropout=0.0, z_seq=None, noise=0.0):
        input_images, input_states = inputs
        bsize = input_images.size(0)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)
        ploss = Variable(torch.zeros(1))
        ploss2 = Variable(torch.zeros(1))
        if self.use_cuda:
            ploss = ploss.cuda()
            ploss2 = ploss2.cuda()

        pred_images, pred_states, pred_costs = [], [], []
        z_list = []

        z = None
        for t in range(npred):
            # encode the inputs (without the action)
            h_x = self.encoder(input_images, input_states)
            if sampling is None:
                # we are training or estimating z distribution
                target_images, target_states, target_costs = targets
                # encode the targets into z
                h_y = self.y_encoder(target_images[:, t].unsqueeze(1).contiguous())
                if random.random() < p_dropout:
                    z = self.z_zero
                else:
                    z = self.z_network(utils.combine(h_x, h_y, self.opt.combine).view(bsize, -1))
                    if self.opt.z_sphere == 1:
                        z = z / torch.norm(z, 2, 1).unsqueeze(1)
                    if save_z:
                        self.save_z(z)
                    if self.opt.beta > 0:
                        assert(False)
                        u = self.prior_network(h_x)
                        ploss = utils.hinge_loss(u, z)
                        '''
                        pi, mu, sigma = self.prior_network(h_x)
                        ploss = utils.mdn_loss_fn(pi, sigma, mu, z)
                        '''
                        if math.isnan(ploss.data[0]):
                            pdb.set_trace()                                                                
            else:
                # we are doing inference
                if z_seq is not None:
                    z = [z_seq[t], None]
                else:
                    z = self.sample_z(bsize, sampling, input_images, input_states, actions[:, t], z, t0=False)

            z_ = z if sampling is None else z[0]
            if noise > 0.0:
                pdb.set_trace()
                z_.data.add_(torch.randn(z_.size()).mul_(noise))

            if hasattr(self, 'action_indep_net'):
                if self.opt.action_indep_net == 1:
                    # loss encouraging independence from a - backpropr to z
                    z_d = z_.clone()
                    z_d.detach()
                    z_d = z_d.detach()
                    loss_s, loss_sa = self.action_indep_net(input_images, input_states, actions[:, t], z_d)
                    ploss += torch.mean((loss_s - loss_sa)**2)
                    # loss to train the prior network - don't backprop to z
                    loss_s_, loss_sa_ = self.action_indep_net(input_images, input_states, actions[:, t], z_d)
                    ploss2 += torch.mean(loss_s_ + loss_sa_)

            z_list.append(z_)
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            a_emb = self.a_encoder(actions[:, t]).view(h_x.size())
            if self.opt.zmult == 0:
                h = utils.combine(h_x, z_exp, self.opt.combine)
                h = utils.combine(h, a_emb, self.opt.combine)
            elif self.opt.zmult == 1:
                h = h_x + a_emb + a_emb * z_exp
            elif self.opt.zmult == 2:
                h = h_x + a_emb - a_emb * z_exp
                
            h = h + self.u_network(h)

            pred_image, pred_state, pred_cost = self.decoder(h)
            if sampling is not None:
                pred_image.detach()
                pred_state.detach()
                pred_cost.detach()
            if self.opt.sigmoid_out == 1:
                pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            else:
                pred_image = torch.clamp(pred_image + input_images[:, -1].unsqueeze(1), min=0, max=1)
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        z_list = torch.stack(z_list, 1)
        return [pred_images, pred_states, pred_costs, z_list], [ploss, ploss2]


    def plan_actions_backprop(self, observation, args, verbose=True, normalize=True, optimize_z=False, optimize_a=True, gamma=0.99):
        input_images, input_states = observation
        input_images = input_images.float().clone()
        input_states = input_states.clone()
        if normalize:
            input_images.div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = Variable(input_images.cuda()).unsqueeze(0)
            input_states = Variable(input_states.cuda()).unsqueeze(0)
        bsize = input_images.size(0)
        # repeat for multiple rollouts
        input_images = input_images.unsqueeze(1).expand(bsize, args.n_rollouts, args.ncond, 3, args.height, args.width)
        input_states = input_states.unsqueeze(1).expand(bsize, args.n_rollouts, args.ncond, 4)
        input_images = input_images.contiguous().view(bsize * args.n_rollouts, args.ncond, 3, args.height, args.width)
        input_states = input_states.contiguous().view(bsize * args.n_rollouts, args.ncond, 4)
        actions = Variable(torch.randn(bsize, args.npred, self.opt.n_actions).cuda().mul_(0.0))
        if optimize_a:
            actions.requires_grad = True
            optimizer_a = optim.Adam([actions], args.lrt)

        Z = self.sample_z(bsize * args.n_rollouts * args.npred, method='fp')[0]
        Z = Z.view(args.npred, bsize * args.n_rollouts, -1)
        Z0 = Z.clone()
        gamma_mask = Variable(torch.from_numpy(numpy.array([[gamma**t, gamma**t] for t in range(args.npred)])).float().cuda()).unsqueeze(0)
        if optimize_z:
            Z.requires_grad = True
            optimizer_z = optim.Adam([Z], args.lrt)
        pred = None
        for i in range(0, args.n_iter):
            optimizer_a.zero_grad()
            self.zero_grad()
            actions_rep = actions.unsqueeze(1).expand(bsize, args.n_rollouts, args.npred, 2).contiguous().view(bsize * args.n_rollouts, args.npred, 2)
            pred, _ = self.forward([input_images, input_states], actions_rep, None, sampling='fp', z_seq=Z)
            costs = pred[2]
            weighted_costs = costs * gamma_mask
            loss = weighted_costs[:, :, 0].mean() + 0.5*weighted_costs[:, :, 1].mean()
            loss.backward()
            if verbose:
                print('[iter {} | mean pred cost = ({:.4f}, {:.4f})], grad = {}'.format(i, costs.data[:, :, 0].mean(), costs.data[:, :, 1].mean(), actions.grad.data.norm()))
            if optimize_a:
                torch.nn.utils.clip_grad_norm([actions], 1)
                optimizer_a.step()
                actions.data.clamp_(min=-2, max=2)
                actions.data[:, :, 1].clamp_(min=-1, max=1)

            if optimize_z:
                torch.nn.utils.clip_grad_norm([Z], 1)
                Z.grad *= -1
                optimizer_z.step()


        # evaluate on new futures
        Z_test = self.sample_z(bsize * args.n_rollouts * args.npred, method='fp')[0]
        Z_test = Z_test.view(args.npred, bsize * args.n_rollouts, -1)
        actions_rep = actions.unsqueeze(1).expand(bsize, args.n_rollouts, args.npred, 2).contiguous().view(bsize * args.n_rollouts, args.npred, 2)
        pred, _ = self.forward([input_images, input_states], actions_rep, None, sampling='fp', z_seq=Z_test)
        costs = pred[2]
        loss_test = costs[:, :, 0].mean() + 0.5*costs[:, :, 1].mean()
        print('\n[pred test cost = {}]\n'.format(loss_test.data[0]))
        
        '''
        # also get predictions using zero actions
        pred_const, _ = self.forward([input_images, input_states], actions.clone().zero_(), None, sampling='fp', z_seq=Z0)
        '''
        pred_const = None
        actions = actions.data.cpu()
        if normalize:
            actions *= self.stats['a_std'].view(1, 1, 2).expand(actions.size())
            actions += self.stats['a_mean'].view(1, 1, 2).expand(actions.size())
        return actions.squeeze().numpy(), pred, pred_const, loss_test.data[0]


    def create_policy_net(self, opt):
        if opt.policy == 'policy-ten':
            self.policy_net = PolicyTEN(opt)
        elif opt.policy == 'policy-vae':
            self.policy_net = PolicyVAE(opt)

    def create_prior_net(self, opt):
        self.prior_net = PriorGaussian(opt, opt.context_dim)

        

    def train_policy_net(self, inputs, targets, targetprop=0, dropout=0.0, save_z=False):
        input_images, input_states = inputs
        target_images, target_states, target_costs = targets
        bsize = input_images.size(0)
        npred = target_images.size(1)
        pred_images, pred_states, pred_costs, pred_actions = [], [], [], []
            
        z, actions_context = None, None
        total_ploss = Variable(torch.zeros(1).cuda())
        self.policy_net.cntr = 0
        for t in range(npred):
            if self.opt.actions_subsample != -1:
                next_images = target_images[:, t:t+self.opt.actions_subsample]
                next_states = target_states[:, t:t+self.opt.actions_subsample]
            else:
                next_images, next_states = None, None
            actions, ploss, _, _, u = self.policy_net(input_images, input_states, next_images, next_states, sample=False, dropout=dropout, save_z=save_z)
            total_ploss += ploss
            # encode the inputs
            h_x = self.encoder(input_images, input_states)
            # encode the targets into z
            h_y = self.y_encoder(target_images[:, t].unsqueeze(1).contiguous())
            z = self.z_network(utils.combine(h_x, h_y, self.opt.combine).view(bsize, -1))
            z_ = z
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)
            a_emb = self.a_encoder(actions).view(h.size())
            h = utils.combine(h, a_emb, self.opt.combine)
            h = h + self.u_network(h)
            pred_image, pred_state, pred_cost = self.decoder(h)
            if self.opt.sigmoid_out == 1:
                pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            else:
                pred_image = torch.clamp(pred_image + input_images[:, -1].unsqueeze(1), min=0, max=1)
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)
            pred_actions.append(actions)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        pred_actions = torch.stack(pred_actions, 1)
        return [pred_images, pred_states, pred_costs, ploss], pred_actions



    def train_prior_net(self, inputs, targets):
        input_images, input_states = inputs
        target_images, target_states, target_costs = targets
        bsize = input_images.size(0)
        npred = target_images.size(1)
        pred_images, pred_states, pred_costs, pred_actions = [], [], [], []
            
        self.policy_net.cntr = 0
        actions, _, _, _, u = self.policy_net(input_images, input_states, target_images, target_states)
        mu, sigma, _ = self.prior_net(input_images, input_states)
        loss = utils.log_pdf(u, mu, sigma)
        return loss.mean()




    # assuming this is for a single input sample
    def compute_action_policy_net(self, inputs, opt, npred=50, n_actions=10, n_futures=5, normalize=True):
        input_images, input_states = inputs
        pred_images, pred_states, pred_costs = [], [], []

        if normalize:
            input_images = input_images.clone().float().div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = Variable(input_images.cuda()).unsqueeze(0)
            input_states = Variable(input_states.cuda()).unsqueeze(0)


        # repeat for multiple rollouts
        bsize = n_futures * n_actions
        input_images = input_images.expand(bsize, opt.ncond, 3, opt.height, opt.width)
        input_states = input_states.expand(bsize, opt.ncond, 4)
        input_images = input_images.contiguous().view(bsize, opt.ncond, 3, opt.height, opt.width)
        input_states = input_states.contiguous().view(bsize, opt.ncond, 4)
            
        Z = self.sample_z(n_futures * npred, method='fp')[0]
        Z = Z.view(npred, 1, n_futures, -1)
        Z = Z.expand(npred, n_actions, n_futures, -1)
        Z0 = Z.clone()

        # sample initial actions: we will choose to execute one of these at the end
        actions_init = self.policy_net(input_images[0].unsqueeze(0), input_states[0].unsqueeze(0), sample=True, normalize=False, n_samples=n_actions)
        actions = actions.clone().unsqueeze(0).expand(n_futures, n_actions, 2)
        for t in range(npred):
            # encode the inputs
            actions = self.policy_net(input_images, input_states)
            h_x = self.encoder(input_images, input_states, actions)
            z_ = Z[t]
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        return [pred_images, pred_states, pred_costs], None


        
    def intype(self, t):
        if t == 'gpu':
            self.cuda()
            self.z_zero = self.z_zero.cuda()
            self.use_cuda = True
            if len(self.p_z) > 0:
                self.p_z = self.p_z.cuda()
        elif t == 'cpu':
            self.cpu()
            self.use_cuda = False
            self.z_zero = self.z_zero.cpu()
            if len(self.p_z) > 0:
                self.p_z = self.p_z.cpu()





















# this version adds the actions *after* the z variables
class FwdCNN_VAE3(nn.Module):
    def __init__(self, opt, mfile=''):
        super(FwdCNN_VAE3, self).__init__()
        self.opt = opt

        if mfile == '':
            self.encoder = encoder(opt, 0, opt.ncond)
            self.decoder = decoder(opt)
        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)['model']
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.a_encoder = pretrained_model.a_encoder
            self.u_network = pretrained_model.u_network
            self.encoder.n_inputs = opt.ncond
            self.decoder.n_out = 1

        self.y_encoder = encoder(opt, 0, 1, states=False)

        self.a_encoder = nn.Sequential(
            nn.Linear(self.opt.n_actions, self.opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.opt.hidden_size)
        )

        self.u_network = u_network(opt)

        self.z_network = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, 2*opt.nz)
        )

        if self.opt.model == 'fwd-cnn-vae3-lp':
            self.z_network_prior = nn.Sequential(
                nn.Linear(opt.hidden_size, opt.nfeature),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(opt.nfeature, opt.nfeature),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(opt.nfeature, 2*opt.nz)
            )


        self.z_zero = Variable(torch.zeros(self.opt.batch_size, self.opt.nz))
        self.z_expander = nn.Linear(opt.nz, opt.hidden_size)

    def reparameterize(self, mu, logvar, sample):
        if self.training or sample:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def sample_z(self, h_x):
        bsize = h_x.size(0)
        if self.opt.model == 'fwd-cnn-vae3-fp':
            z = Variable(torch.randn(bsize, self.opt.nz).cuda())
        elif self.opt.model == 'fwd-cnn-vae3-lp':
            mu_logvar_prior = self.z_network_prior(h_x.view(bsize, -1)).view(bsize, 2, self.opt.nz)
            mu_prior = mu_logvar_prior[:, 0]
            logvar_prior = mu_logvar_prior[:, 1]
            z = self.reparameterize(mu_prior, logvar_prior, True)
        return z
            

    def forward(self, inputs, actions, targets, save_z = False, sampling=None, p_dropout=0.0, z_seq=None, noise=None):
        input_images, input_states = inputs
        bsize = input_images.size(0)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)
        ploss = Variable(torch.zeros(1))
        ploss2 = Variable(torch.zeros(1))
        if self.use_cuda:
            ploss = ploss.cuda()
            ploss2 = ploss2.cuda()

        pred_images, pred_states, pred_costs = [], [], []
        z_list = []

        z = None
        for t in range(npred):
            # encode the inputs (without the action)
            h_x = self.encoder(input_images, input_states)
            if sampling is None:
                # we are training or estimating z distribution
                target_images, target_states, target_costs = targets
                # encode the targets into z
                h_y = self.y_encoder(target_images[:, t].unsqueeze(1).contiguous())
                if random.random() < p_dropout:
                    z = Variable(self.sample_z(h_x).data)
                else:
                    mu_logvar = self.z_network(utils.combine(h_x, h_y, self.opt.combine).view(bsize, -1)).view(bsize, 2, self.opt.nz)
                    mu = mu_logvar[:, 0]
                    logvar = mu_logvar[:, 1]
                    z = self.reparameterize(mu, logvar, True)
                    logvar = torch.clamp(logvar, max = 4) # this can go to inf when taking exp(), so clamp it
                    if self.opt.model == 'fwd-cnn-vae3-fp':
                        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        kld /= bsize
                    elif self.opt.model == 'fwd-cnn-vae3-lp':
                        mu_logvar_prior = self.z_network_prior(h_x.view(bsize, -1)).view(bsize, 2, self.opt.nz)
                        mu_prior = mu_logvar_prior[:, 0]
                        logvar_prior = mu_logvar_prior[:, 1]
                        kld = utils.kl_criterion(mu, logvar, mu_prior, logvar_prior)
                    ploss += kld
            else:
                z = self.sample_z(h_x)

            z_list.append(z)
            z_exp = self.z_expander(z).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp, self.opt.combine)
            a_emb = self.a_encoder(actions[:, t]).view(h.size())
            h = utils.combine(h, a_emb, self.opt.combine)
            h = h + self.u_network(h)

            pred_image, pred_state, pred_cost = self.decoder(h)
            if sampling is not None:
                pred_image.detach()
                pred_state.detach()
                pred_cost.detach()
            if self.opt.sigmoid_out == 1:
                pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            else:
                pred_image = torch.clamp(pred_image + input_images[:, -1].unsqueeze(1), min=0, max=1)
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        z_list = torch.stack(z_list, 1)
        return [pred_images, pred_states, pred_costs, z_list], [ploss, ploss2]


    def plan_actions_backprop(self, observation, args, verbose=True, normalize=True, optimize_z=False, optimize_a=True, gamma=0.99):
        input_images, input_states = observation
        input_images = input_images.float().clone()
        input_states = input_states.clone()
        if normalize:
            input_images.div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = Variable(input_images.cuda()).unsqueeze(0)
            input_states = Variable(input_states.cuda()).unsqueeze(0)
        bsize = input_images.size(0)
        # repeat for multiple rollouts
        input_images = input_images.unsqueeze(1).expand(bsize, args.n_rollouts, args.ncond, 3, args.height, args.width)
        input_states = input_states.unsqueeze(1).expand(bsize, args.n_rollouts, args.ncond, 4)
        input_images = input_images.contiguous().view(bsize * args.n_rollouts, args.ncond, 3, args.height, args.width)
        input_states = input_states.contiguous().view(bsize * args.n_rollouts, args.ncond, 4)
        actions = Variable(torch.randn(bsize, args.npred, self.opt.n_actions).cuda().mul_(0.0))
        if optimize_a:
            actions.requires_grad = True
            optimizer_a = optim.Adam([actions], args.lrt)

        Z = self.sample_z(bsize * args.n_rollouts * args.npred, method='fp')[0]
        Z = Z.view(args.npred, bsize * args.n_rollouts, -1)
        Z0 = Z.clone()
        gamma_mask = Variable(torch.from_numpy(numpy.array([[gamma**t, gamma**t] for t in range(args.npred)])).float().cuda()).unsqueeze(0)
        if optimize_z:
            Z.requires_grad = True
            optimizer_z = optim.Adam([Z], args.lrt)
        pred = None
        for i in range(0, args.n_iter):
            optimizer_a.zero_grad()
            self.zero_grad()
            actions_rep = actions.unsqueeze(1).expand(bsize, args.n_rollouts, args.npred, 2).contiguous().view(bsize * args.n_rollouts, args.npred, 2)
            pred, _ = self.forward([input_images, input_states], actions_rep, None, sampling='fp', z_seq=Z)
            costs = pred[2]
            weighted_costs = costs * gamma_mask
            loss = weighted_costs[:, :, 0].mean() + 0.5*weighted_costs[:, :, 1].mean()
            loss.backward()
            if verbose:
                print('[iter {} | mean pred cost = ({:.4f}, {:.4f})], grad = {}'.format(i, costs.data[:, :, 0].mean(), costs.data[:, :, 1].mean(), actions.grad.data.norm()))
            if optimize_a:
                torch.nn.utils.clip_grad_norm([actions], 1)
                optimizer_a.step()
                actions.data.clamp_(min=-2, max=2)
                actions.data[:, :, 1].clamp_(min=-1, max=1)

            if optimize_z:
                torch.nn.utils.clip_grad_norm([Z], 1)
                Z.grad *= -1
                optimizer_z.step()


        # evaluate on new futures
        Z_test = self.sample_z(bsize * args.n_rollouts * args.npred, method='fp')[0]
        Z_test = Z_test.view(args.npred, bsize * args.n_rollouts, -1)
        actions_rep = actions.unsqueeze(1).expand(bsize, args.n_rollouts, args.npred, 2).contiguous().view(bsize * args.n_rollouts, args.npred, 2)
        pred, _ = self.forward([input_images, input_states], actions_rep, None, sampling='fp', z_seq=Z_test)
        costs = pred[2]
        loss_test = costs[:, :, 0].mean() + 0.5*costs[:, :, 1].mean()
        print('\n[pred test cost = {}]\n'.format(loss_test.data[0]))
        
        '''
        # also get predictions using zero actions
        pred_const, _ = self.forward([input_images, input_states], actions.clone().zero_(), None, sampling='fp', z_seq=Z0)
        '''
        pred_const = None
        actions = actions.data.cpu()
        if normalize:
            actions *= self.stats['a_std'].view(1, 1, 2).expand(actions.size())
            actions += self.stats['a_mean'].view(1, 1, 2).expand(actions.size())
        return actions.squeeze().numpy(), pred, pred_const, loss_test.data[0]


    def create_policy_net(self, opt):
        if opt.policy == 'policy-ten':
            self.policy_net = PolicyTEN(opt)
        elif opt.policy == 'policy-vae':
            self.policy_net = PolicyVAE(opt)

    def create_prior_net(self, opt):
        self.prior_net = PriorGaussian(opt, opt.context_dim)

        

    def train_policy_net(self, inputs, targets, targetprop=0, dropout=0.0, save_z=False):
        input_images, input_states = inputs
        target_images, target_states, target_costs = targets
        bsize = input_images.size(0)
        npred = target_images.size(1)
        pred_images, pred_states, pred_costs, pred_actions = [], [], [], []
            
        z, actions_context = None, None
        total_ploss = Variable(torch.zeros(1).cuda())
        self.policy_net.cntr = 0
        for t in range(npred):
            if self.opt.actions_subsample != -1:
                next_images = target_images[:, t:t+self.opt.actions_subsample]
                next_states = target_states[:, t:t+self.opt.actions_subsample]
            else:
                next_images, next_states = None, None
            actions, ploss, _, _, u = self.policy_net(input_images, input_states, next_images, next_states, sample=False, dropout=dropout, save_z=save_z)
            total_ploss += ploss
            # encode the inputs
            h_x = self.encoder(input_images, input_states)
            # encode the targets into z
            h_y = self.y_encoder(target_images[:, t].unsqueeze(1).contiguous())
            z = self.z_network(utils.combine(h_x, h_y, self.opt.combine).view(bsize, -1))
            z_ = z
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)
            a_emb = self.a_encoder(actions).view(h.size())
            h = utils.combine(h, a_emb, self.opt.combine)
            h = h + self.u_network(h)
            pred_image, pred_state, pred_cost = self.decoder(h)
            if self.opt.sigmoid_out == 1:
                pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            else:
                pred_image = torch.clamp(pred_image + input_images[:, -1].unsqueeze(1), min=0, max=1)
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)
            pred_actions.append(actions)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        pred_actions = torch.stack(pred_actions, 1)
        return [pred_images, pred_states, pred_costs, ploss], pred_actions



    def train_prior_net(self, inputs, targets):
        input_images, input_states = inputs
        target_images, target_states, target_costs = targets
        bsize = input_images.size(0)
        npred = target_images.size(1)
        pred_images, pred_states, pred_costs, pred_actions = [], [], [], []
            
        self.policy_net.cntr = 0
        actions, _, _, _, u = self.policy_net(input_images, input_states, target_images, target_states)
        mu, sigma, _ = self.prior_net(input_images, input_states)
        loss = utils.log_pdf(u, mu, sigma)
        return loss.mean()




    # assuming this is for a single input sample
    def compute_action_policy_net(self, inputs, opt, npred=50, n_actions=10, n_futures=5, normalize=True):
        input_images, input_states = inputs
        pred_images, pred_states, pred_costs = [], [], []

        if normalize:
            input_images = input_images.clone().float().div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = Variable(input_images.cuda()).unsqueeze(0)
            input_states = Variable(input_states.cuda()).unsqueeze(0)


        # repeat for multiple rollouts
        bsize = n_futures * n_actions
        input_images = input_images.expand(bsize, opt.ncond, 3, opt.height, opt.width)
        input_states = input_states.expand(bsize, opt.ncond, 4)
        input_images = input_images.contiguous().view(bsize, opt.ncond, 3, opt.height, opt.width)
        input_states = input_states.contiguous().view(bsize, opt.ncond, 4)
            
        Z = self.sample_z(n_futures * npred, method='fp')[0]
        Z = Z.view(npred, 1, n_futures, -1)
        Z = Z.expand(npred, n_actions, n_futures, -1)
        Z0 = Z.clone()

        # sample initial actions: we will choose to execute one of these at the end
        actions_init = self.policy_net(input_images[0].unsqueeze(0), input_states[0].unsqueeze(0), sample=True, normalize=False, n_samples=n_actions)
        actions = actions.clone().unsqueeze(0).expand(n_futures, n_actions, 2)
        for t in range(npred):
            # encode the inputs
            actions = self.policy_net(input_images, input_states)
            h_x = self.encoder(input_images, input_states, actions)
            z_ = Z[t]
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        return [pred_images, pred_states, pred_costs], None


        
    def intype(self, t):
        if t == 'gpu':
            self.cuda()
            self.z_zero = self.z_zero.cuda()
            self.use_cuda = True
        elif t == 'cpu':
            self.cpu()
            self.use_cuda = False
            self.z_zero = self.z_zero.cpu()











# this version adds the actions *after* the z variables
class FwdCNN_VAE3_STN(nn.Module):
    def __init__(self, opt, mfile=''):
        super(FwdCNN_VAE3_STN, self).__init__()
        self.opt = opt

        if mfile == '':
            self.encoder = encoder(opt, 0, opt.ncond)
            self.decoder = decoder(opt)
        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)['model']
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.a_encoder = pretrained_model.a_encoder
            self.u_network = pretrained_model.u_network
            self.encoder.n_inputs = opt.ncond
            self.decoder.n_out = 1

        self.y_encoder = encoder(opt, 0, 1, states=False)

        self.a_encoder = nn.Sequential(
            nn.Linear(self.opt.n_actions, self.opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.opt.hidden_size)
        )

        self.u_network = u_network(opt)

        self.z_network = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, 2*opt.nz)
        )

        if self.opt.model == 'fwd-cnn-vae3-lp':
            self.z_network_prior = nn.Sequential(
                nn.Linear(opt.hidden_size, opt.nfeature),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(opt.nfeature, opt.nfeature),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(opt.nfeature, 2*opt.nz)
            )


        self.z_zero = Variable(torch.zeros(self.opt.batch_size, self.opt.nz))
        self.z_expander = nn.Linear(opt.nz, opt.hidden_size)

    def reparameterize(self, mu, logvar, sample):
        if self.training or sample:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def sample_z(self, h_x):
        bsize = h_x.size(0)
        if self.opt.model == 'fwd-cnn-vae3-fp':
            z = Variable(torch.randn(bsize, self.opt.nz).cuda())
        elif self.opt.model == 'fwd-cnn-vae3-lp':
            mu_logvar_prior = self.z_network_prior(h_x.view(bsize, -1)).view(bsize, 2, self.opt.nz)
            mu_prior = mu_logvar_prior[:, 0]
            logvar_prior = mu_logvar_prior[:, 1]
            z = self.reparameterize(mu_prior, logvar_prior, True)
        return z
            

    def forward(self, inputs, actions, targets, save_z = False, sampling=None, p_dropout=0.0, z_seq=None, noise=None):
        input_images, input_states = inputs
        bsize = input_images.size(0)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)
        ploss = Variable(torch.zeros(1))
        ploss2 = Variable(torch.zeros(1))
        if self.use_cuda:
            ploss = ploss.cuda()
            ploss2 = ploss2.cuda()

        pred_images, pred_states, pred_costs = [], [], []
        z_list = []

        z = None
        for t in range(npred):
            # encode the inputs (without the action)
            h_x = self.encoder(input_images, input_states)
            if sampling is None:
                # we are training or estimating z distribution
                target_images, target_states, target_costs = targets
                # encode the targets into z
                h_y = self.y_encoder(target_images[:, t].unsqueeze(1).contiguous())
                if random.random() < p_dropout:
                    z = Variable(self.sample_z(h_x).data)
                else:
                    mu_logvar = self.z_network(utils.combine(h_x, h_y, self.opt.combine).view(bsize, -1)).view(bsize, 2, self.opt.nz)
                    mu = mu_logvar[:, 0]
                    logvar = mu_logvar[:, 1]
                    z = self.reparameterize(mu, logvar, True)
                    logvar = torch.clamp(logvar, max = 4) # this can go to inf when taking exp(), so clamp it
                    if self.opt.model == 'fwd-cnn-vae3-fp':
                        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        kld /= bsize
                    elif self.opt.model == 'fwd-cnn-vae3-lp':
                        mu_logvar_prior = self.z_network_prior(h_x.view(bsize, -1)).view(bsize, 2, self.opt.nz)
                        mu_prior = mu_logvar_prior[:, 0]
                        logvar_prior = mu_logvar_prior[:, 1]
                        kld = utils.kl_criterion(mu, logvar, mu_prior, logvar_prior)
                    ploss += kld
            else:
                z = self.sample_z(h_x)

            z_list.append(z)
            z_exp = self.z_expander(z).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp, self.opt.combine)
            a_emb = self.a_encoder(actions[:, t]).view(h.size())
            h = utils.combine(h, a_emb, self.opt.combine)
            h = h + self.u_network(h)

            pred_image, pred_state, pred_cost = self.decoder(h)



            if sampling is not None:
                pred_image.detach()
                pred_state.detach()
                pred_cost.detach()
            if self.opt.sigmoid_out == 1:
                pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            else:
                pred_image = torch.clamp(pred_image + input_images[:, -1].unsqueeze(1), min=0, max=1)
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        z_list = torch.stack(z_list, 1)
        return [pred_images, pred_states, pred_costs, z_list], [ploss, ploss2]


    def plan_actions_backprop(self, observation, args, verbose=True, normalize=True, optimize_z=False, optimize_a=True, gamma=0.99):
        input_images, input_states = observation
        input_images = input_images.float().clone()
        input_states = input_states.clone()
        if normalize:
            input_images.div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = Variable(input_images.cuda()).unsqueeze(0)
            input_states = Variable(input_states.cuda()).unsqueeze(0)
        bsize = input_images.size(0)
        # repeat for multiple rollouts
        input_images = input_images.unsqueeze(1).expand(bsize, args.n_rollouts, args.ncond, 3, args.height, args.width)
        input_states = input_states.unsqueeze(1).expand(bsize, args.n_rollouts, args.ncond, 4)
        input_images = input_images.contiguous().view(bsize * args.n_rollouts, args.ncond, 3, args.height, args.width)
        input_states = input_states.contiguous().view(bsize * args.n_rollouts, args.ncond, 4)
        actions = Variable(torch.randn(bsize, args.npred, self.opt.n_actions).cuda().mul_(0.0))
        if optimize_a:
            actions.requires_grad = True
            optimizer_a = optim.Adam([actions], args.lrt)

        Z = self.sample_z(bsize * args.n_rollouts * args.npred, method='fp')[0]
        Z = Z.view(args.npred, bsize * args.n_rollouts, -1)
        Z0 = Z.clone()
        gamma_mask = Variable(torch.from_numpy(numpy.array([[gamma**t, gamma**t] for t in range(args.npred)])).float().cuda()).unsqueeze(0)
        if optimize_z:
            Z.requires_grad = True
            optimizer_z = optim.Adam([Z], args.lrt)
        pred = None
        for i in range(0, args.n_iter):
            optimizer_a.zero_grad()
            self.zero_grad()
            actions_rep = actions.unsqueeze(1).expand(bsize, args.n_rollouts, args.npred, 2).contiguous().view(bsize * args.n_rollouts, args.npred, 2)
            pred, _ = self.forward([input_images, input_states], actions_rep, None, sampling='fp', z_seq=Z)
            costs = pred[2]
            weighted_costs = costs * gamma_mask
            loss = weighted_costs[:, :, 0].mean() + 0.5*weighted_costs[:, :, 1].mean()
            loss.backward()
            if verbose:
                print('[iter {} | mean pred cost = ({:.4f}, {:.4f})], grad = {}'.format(i, costs.data[:, :, 0].mean(), costs.data[:, :, 1].mean(), actions.grad.data.norm()))
            if optimize_a:
                torch.nn.utils.clip_grad_norm([actions], 1)
                optimizer_a.step()
                actions.data.clamp_(min=-2, max=2)
                actions.data[:, :, 1].clamp_(min=-1, max=1)

            if optimize_z:
                torch.nn.utils.clip_grad_norm([Z], 1)
                Z.grad *= -1
                optimizer_z.step()


        # evaluate on new futures
        Z_test = self.sample_z(bsize * args.n_rollouts * args.npred, method='fp')[0]
        Z_test = Z_test.view(args.npred, bsize * args.n_rollouts, -1)
        actions_rep = actions.unsqueeze(1).expand(bsize, args.n_rollouts, args.npred, 2).contiguous().view(bsize * args.n_rollouts, args.npred, 2)
        pred, _ = self.forward([input_images, input_states], actions_rep, None, sampling='fp', z_seq=Z_test)
        costs = pred[2]
        loss_test = costs[:, :, 0].mean() + 0.5*costs[:, :, 1].mean()
        print('\n[pred test cost = {}]\n'.format(loss_test.data[0]))
        
        '''
        # also get predictions using zero actions
        pred_const, _ = self.forward([input_images, input_states], actions.clone().zero_(), None, sampling='fp', z_seq=Z0)
        '''
        pred_const = None
        actions = actions.data.cpu()
        if normalize:
            actions *= self.stats['a_std'].view(1, 1, 2).expand(actions.size())
            actions += self.stats['a_mean'].view(1, 1, 2).expand(actions.size())
        return actions.squeeze().numpy(), pred, pred_const, loss_test.data[0]


    def create_policy_net(self, opt):
        if opt.policy == 'policy-ten':
            self.policy_net = PolicyTEN(opt)
        elif opt.policy == 'policy-vae':
            self.policy_net = PolicyVAE(opt)

    def create_prior_net(self, opt):
        self.prior_net = PriorGaussian(opt, opt.context_dim)

        

    def train_policy_net(self, inputs, targets, targetprop=0, dropout=0.0, save_z=False):
        input_images, input_states = inputs
        target_images, target_states, target_costs = targets
        bsize = input_images.size(0)
        npred = target_images.size(1)
        pred_images, pred_states, pred_costs, pred_actions = [], [], [], []
            
        z, actions_context = None, None
        total_ploss = Variable(torch.zeros(1).cuda())
        self.policy_net.cntr = 0
        for t in range(npred):
            if self.opt.actions_subsample != -1:
                next_images = target_images[:, t:t+self.opt.actions_subsample]
                next_states = target_states[:, t:t+self.opt.actions_subsample]
            else:
                next_images, next_states = None, None
            actions, ploss, _, _, u = self.policy_net(input_images, input_states, next_images, next_states, sample=False, dropout=dropout, save_z=save_z)
            total_ploss += ploss
            # encode the inputs
            h_x = self.encoder(input_images, input_states)
            # encode the targets into z
            h_y = self.y_encoder(target_images[:, t].unsqueeze(1).contiguous())
            z = self.z_network(utils.combine(h_x, h_y, self.opt.combine).view(bsize, -1))
            z_ = z
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)
            a_emb = self.a_encoder(actions).view(h.size())
            h = utils.combine(h, a_emb, self.opt.combine)
            h = h + self.u_network(h)
            pred_image, pred_state, pred_cost = self.decoder(h)
            if self.opt.sigmoid_out == 1:
                pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            else:
                pred_image = torch.clamp(pred_image + input_images[:, -1].unsqueeze(1), min=0, max=1)
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)
            pred_actions.append(actions)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        pred_actions = torch.stack(pred_actions, 1)
        return [pred_images, pred_states, pred_costs, ploss], pred_actions



    def train_prior_net(self, inputs, targets):
        input_images, input_states = inputs
        target_images, target_states, target_costs = targets
        bsize = input_images.size(0)
        npred = target_images.size(1)
        pred_images, pred_states, pred_costs, pred_actions = [], [], [], []
            
        self.policy_net.cntr = 0
        actions, _, _, _, u = self.policy_net(input_images, input_states, target_images, target_states)
        mu, sigma, _ = self.prior_net(input_images, input_states)
        loss = utils.log_pdf(u, mu, sigma)
        return loss.mean()




    # assuming this is for a single input sample
    def compute_action_policy_net(self, inputs, opt, npred=50, n_actions=10, n_futures=5, normalize=True):
        input_images, input_states = inputs
        pred_images, pred_states, pred_costs = [], [], []

        if normalize:
            input_images = input_images.clone().float().div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = Variable(input_images.cuda()).unsqueeze(0)
            input_states = Variable(input_states.cuda()).unsqueeze(0)


        # repeat for multiple rollouts
        bsize = n_futures * n_actions
        input_images = input_images.expand(bsize, opt.ncond, 3, opt.height, opt.width)
        input_states = input_states.expand(bsize, opt.ncond, 4)
        input_images = input_images.contiguous().view(bsize, opt.ncond, 3, opt.height, opt.width)
        input_states = input_states.contiguous().view(bsize, opt.ncond, 4)
            
        Z = self.sample_z(n_futures * npred, method='fp')[0]
        Z = Z.view(npred, 1, n_futures, -1)
        Z = Z.expand(npred, n_actions, n_futures, -1)
        Z0 = Z.clone()

        # sample initial actions: we will choose to execute one of these at the end
        actions_init = self.policy_net(input_images[0].unsqueeze(0), input_states[0].unsqueeze(0), sample=True, normalize=False, n_samples=n_actions)
        actions = actions.clone().unsqueeze(0).expand(n_futures, n_actions, 2)
        for t in range(npred):
            # encode the inputs
            actions = self.policy_net(input_images, input_states)
            h_x = self.encoder(input_images, input_states, actions)
            z_ = Z[t]
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        return [pred_images, pred_states, pred_costs], None


        
    def intype(self, t):
        if t == 'gpu':
            self.cuda()
            self.z_zero = self.z_zero.cuda()
            self.use_cuda = True
        elif t == 'cpu':
            self.cpu()
            self.use_cuda = False
            self.z_zero = self.z_zero.cpu()



























# forward VAE model with a fixed prior
class FwdCNN_VAE_FP(nn.Module):
    def __init__(self, opt, mfile=''):
        super(FwdCNN_VAE_FP, self).__init__()
        self.opt = opt
        if mfile == '':
            self.encoder = encoder(opt, opt.n_actions, opt.ncond)
            self.decoder = decoder(opt)
        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.encoder.n_inputs = opt.ncond


        self.y_encoder = encoder(opt, opt.n_actions, opt.ncond + 1)
        self.z_network = z_network_gaussian(opt)
        self.z_expander = z_expander(opt, 1)

    def forward(self, inputs, actions, targets, sampling=None, z_seq=None):
        input_images, input_states = inputs
        bsize = input_images.size(0)
        npred = actions.size(1)
        pred_images, pred_states, pred_costs = [], [], []

        kld = Variable(torch.zeros(1))
        if self.use_cuda:
            kld = kld.cuda()

        pred = []
        for t in range(npred):
            h_x = self.encoder(input_images, input_states, actions[:, t])
            if sampling is None:
                # we are training
                target_images, target_states, target_costs = targets
                h_y = self.y_encoder(torch.cat((input_images, target_images[:, t].unsqueeze(1).contiguous()), 1), input_states, actions[:, t])
                z, mu, logvar = self.z_network(h_y, sample=True)
                logvar = torch.clamp(logvar, max = 4) # this can go to inf when taking exp(), so clamp it
                kld_t = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kld_t /= bsize
                kld += kld_t
                z_exp = self.z_expander(z)
            else:
                # we are generating samples
                if z_seq is not None:
                    z = Variable(z_seq[t].cuda())
                else:
                    z = Variable(torch.randn(bsize, self.opt.nz).cuda())
                z_exp = self.z_expander(z)

            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)

            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        kld /= npred
        return [pred_images, pred_states, pred_costs], kld


    def plan_actions_backprop(self, observation, args, verbose=True, normalize=True):
        input_images, input_states = observation
        input_images = input_images.float().clone()
        input_states = input_states.clone()
        if normalize:
            input_images.div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
        input_images = Variable(input_images.cuda()).unsqueeze(0)
        input_states = Variable(input_states.cuda()).unsqueeze(0)
        bsize = input_images.size(0)
        # repeat for multiple rollouts
        actions = Variable(torch.randn(bsize, args.npred, self.opt.n_actions).cuda().mul_(0.01), requires_grad=True)
        optimizer_a = optim.SGD([actions], args.lrt)
        Z = torch.randn(args.npred, 1, self.opt.nz).cuda()
        for i in range(0, args.n_iter):
            optimizer_a.zero_grad()
            self.zero_grad()
            pred, _ = self.forward([input_images, input_states], actions, None, sampling='vae', z_seq=None)
            costs = pred[2]
            loss = costs[:, :, 0].mean() + 0.5*costs[:, :, 1].mean()
            loss.backward()
            if verbose:
                print('[iter {} | mean pred cost = {}], grad = {}'.format(i, loss.data[0], actions.grad.data.norm()))
            torch.nn.utils.clip_grad_norm([actions], 0.1)
            optimizer_a.step()
            actions.data.clamp_(min=-2, max=2)
            actions.data[:, :, 1].clamp_(min=-1, max=1)
        actions = actions.data.cpu()
        actions *= self.stats['a_std'].view(1, 1, 2).expand(actions.size())
        actions += self.stats['a_mean'].view(1, 1, 2).expand(actions.size())
        return actions.squeeze().numpy()


        

    def intype(self, t):
        if t == 'gpu':
            self.cuda()
            self.use_cuda = True
        elif t == 'cpu':
            self.cpu()
            self.use_cuda = False











# forward VAE model with a learned prior
class FwdCNN_VAE_LP(nn.Module):
    def __init__(self, opt, mfile=''):
        super(FwdCNN_VAE_LP, self).__init__()
        self.opt = opt
        if mfile == '':
            self.encoder = encoder(opt, opt.n_actions, opt.ncond)
            self.decoder = decoder(opt)
        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.encoder.n_inputs = opt.ncond

        # we do not include states in the future encoder, as they can be 
        # too easily reconstructed 
        self.y_encoder = encoder(opt, opt.n_actions, opt.ncond + 1)
        self.prior_encoder = encoder(opt, opt.n_actions, opt.ncond)
        self.z_network = z_network_gaussian(opt)
        self.q_network = z_network_gaussian(opt)
        self.z_expander = z_expander(opt, 1)

    def forward(self, inputs, actions, targets, sampling=None):
        input_images, input_states = inputs
        bsize = input_images.size(0)
        npred = actions.size(1)
        pred_images, pred_states, pred_costs = [], [], []

        kld = Variable(torch.zeros(1))
        if self.use_cuda:
            kld = kld.cuda()

        pred_images, pred_states, pred_costs = [], [], []
        for t in range(npred):
            h_x = self.encoder(input_images, input_states, actions[:, t])
            if numpy.isnan(h_x[0][0][0][0].data[0]):
                pdb.set_trace()
            if sampling is None:
                # we are training
                target_images, target_states, target_costs = targets
                h_y = self.y_encoder(torch.cat((input_images, target_images[:, t].unsqueeze(1).contiguous()), 1), input_states, actions[:, t])
                z1, mu1, logvar1 = self.z_network(h_y, sample=True)
                z2, mu2, logvar2 = self.q_network(self.prior_encoder(input_images, input_states, actions[:, t]))
                sigma1 = logvar1.mul(0.5).exp()
                sigma2 = logvar2.mul(0.5).exp()
                # not sure why, but the learned prior VAE model is unstable to train 
                # it has very high gradients every now and then, and sometimes
                # gives NaN values. Clamping the z vectors and the KL loss helps a bit, 
                # but still doesn't work very well. 

                z1 = torch.clamp(z1, min=-100, max=100)
                z2 = torch.clamp(z2, min=-100, max=100)
                if numpy.isnan(z1.norm().data[0]):
                    pdb.set_trace()
                kld_t = torch.log(sigma2/sigma1 + 1e-5) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(1e-5 + 2*torch.exp(logvar2)) - 1/2
                kld_t = torch.clamp(kld_t, max=50)
                kld_t = torch.sum(kld_t) / bsize
                kld += kld_t
                z_exp = self.z_expander(z1)
#                if numpy.isnan(kld_t.data[0]) or kld_t.data[0] > 40:
#                    pdb.set_trace()
            else:
                # we are generating samples
                z, _, _ = self.q_network(h_x, sample=True)
                z_exp = self.z_expander(z)

            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)

            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        kld /= npred
        return [pred_images, pred_states, pred_costs], kld


    def intype(self, t):
        if t == 'gpu':
            self.cuda()
            self.use_cuda = True
        elif t == 'cpu':
            self.cpu()
            self.use_cuda = False




class LSTMCritic(nn.Module):
    def __init__(self, opt, batch_size=None):
        super(LSTMCritic, self).__init__()
        self.opt = copy.deepcopy(opt)
        self.opt.ncond = 1
        if batch_size is None:
            batch_size = opt.batch_size
        self.encoder = encoder(self.opt, 0, 1, state_input_size=6)
        self.input_size = opt.nfeature*opt.h_height*opt.h_width
        self.lstm = nn.LSTM(self.input_size, opt.nhidden)
        self.classifier = nn.Linear(opt.nhidden, 1)
        self.hidden = (Variable(torch.randn(1, batch_size, opt.nhidden).cuda()),
                       Variable(torch.randn(1, batch_size, opt.nhidden).cuda()))


    def forward(self, inputs):
        input_images, input_states = inputs
        input_images.detach()
        input_states.detach()        
        bsize = input_images.size(0)
        self.hidden = (Variable(torch.randn(1, bsize, self.opt.nhidden).cuda()),Variable(torch.randn(1, bsize, self.opt.nhidden).cuda()))
        T = input_images.size(1)
        for t in range(T):
            enc = self.encoder(input_images[:, t].contiguous(), input_states[:, t].contiguous())
            out, self.hidden = self.lstm(enc.view(1, bsize, -1), self.hidden)
        logits = self.classifier(out.squeeze())
        return logits.squeeze()



#######################################
# Policy Networks
#######################################

# deterministic CNN model
class PolicyCNN(nn.Module):
    def __init__(self, opt):
        super(PolicyCNN, self).__init__()
        self.opt = opt
        self.encoder = encoder(opt)
        self.hsize = opt.nfeature*self.opt.h_height*self.opt.h_width
        self.fc = nn.Sequential(
            nn.Linear(self.hsize + opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.npred*opt.n_actions)
        )

    def forward(self, state_images, states, actions):
        bsize = state_images.size(0)
        h = self.encoder(state_images, states)
        a = self.fc(h)
        a = a.view(bsize, self.opt.npred, self.opt.n_actions)
        return a, Variable(torch.zeros(1))


    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()



# Stochastic Policy, output is a diagonal Gaussian and learning
# uses the reparameterization trick. 
class StochasticPolicy(nn.Module):
    def __init__(self, opt, context_dim=0, output_dim=None):
        super(StochasticPolicy, self).__init__()
        self.opt = opt
        self.encoder = encoder(opt, 0, opt.ncond)
        self.hsize = opt.nfeature*self.opt.h_height*self.opt.h_width
        self.n_outputs = opt.n_actions if output_dim is None else output_dim
        self.proj = nn.Linear(self.hsize, opt.n_hidden)
        self.context_dim = context_dim

        self.fc = nn.Sequential(
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden)
        )

        if context_dim > 0:
            self.context_encoder = nn.Sequential(
                nn.Linear(context_dim, opt.n_hidden), 
                nn.ReLU(), 
                nn.Linear(opt.n_hidden, opt.n_hidden), 
                nn.ReLU(), 
                nn.Linear(opt.n_hidden, opt.n_hidden)
            )

        self.mu_net = nn.Linear(opt.n_hidden, self.n_outputs)
        self.logvar_net = nn.Linear(opt.n_hidden, self.n_outputs)


    def forward(self, state_images, states, context=None, sample=True, normalize_inputs=False, normalize_outputs=False, n_samples=1):

        if normalize_inputs:
            state_images = state_images.clone().float().div_(255.0)
            states -= self.stats['s_mean'].view(1, 4).expand(states.size())
            states /= self.stats['s_std'].view(1, 4).expand(states.size())
            state_images = Variable(state_images.cuda()).unsqueeze(0)
            states = Variable(states.cuda()).unsqueeze(0)


        bsize = state_images.size(0)

        h = self.encoder(state_images, states).view(bsize, self.hsize)
        h = self.proj(h)
        if self.context_dim > 0:
            assert(context is not None)
            h = h + self.context_encoder(context)
        h = self.fc(h)
        mu = self.mu_net(h).view(bsize, self.n_outputs)
        logvar = self.logvar_net(h).view(bsize, self.n_outputs)
        logvar = torch.clamp(logvar, max = 4.0)
        std = logvar.mul(0.5).exp_()
        eps = Variable(torch.randn(bsize, n_samples, self.n_outputs).cuda())
        a = eps * std.view(bsize, 1, self.n_outputs)
        a = a + mu.view(bsize, 1, self.n_outputs)
#        eps = Variable(std.data.new(std.size()).normal_())
#        a = eps.mul(std).add_(mu)


        if normalize_outputs:
            a = a.data
            a.clamp_(-3, 3)
            a *= self.stats['a_std'].view(1, 1, 2).expand(a.size()).cuda()
            a += self.stats['a_mean'].view(1, 1, 2).expand(a.size()).cuda()

        entropy = std.mean()
        return a.squeeze(), entropy, mu, std


    def intype(self, t):
        if t == 'gpu':
            self.cuda()
            if hasattr(self, 'a_mean'):
                self.a_mean = self.a_mean.cuda()
                self.a_std = self.a_std.cuda()
                self.s_mean = self.s_mean.cuda()
                self.s_std = self.s_std.cuda()
        elif t == 'cpu':
            self.cpu()
            if hasattr(self, 'a_mean'):
                self.a_mean = self.a_mean.cpu()
                self.a_std = self.a_std.cpu()
                self.s_mean = self.s_mean.cpu()
                self.s_std = self.s_std.cpu()






class PolicyTEN(nn.Module):
    def __init__(self, opt):
        super(PolicyTEN, self).__init__()
        self.opt = opt
        self.past_encoder = encoder(opt, 0, opt.ncond)
        self.proj_past = nn.Linear(opt.hidden_size, opt.n_hidden)

        self.fc = nn.Sequential(
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden)
        )

        if opt.actions_subsample != -1:
            self.future_encoder = encoder(opt, 0, opt.actions_subsample)
            self.proj_future = nn.Linear(opt.hidden_size, opt.n_hidden)
            self.fc_z = nn.Sequential(
                nn.Linear(opt.n_hidden, opt.n_hidden),
                nn.ReLU(),
                nn.Linear(opt.n_hidden, opt.n_hidden),
                nn.ReLU(),
                nn.Linear(opt.n_hidden, self.opt.context_dim)
            )

            self.z_exp = nn.Sequential(
                nn.Linear(opt.context_dim, opt.n_hidden), 
                nn.ReLU(), 
                nn.Linear(opt.n_hidden, opt.n_hidden), 
                nn.ReLU(), 
                nn.Linear(opt.n_hidden, opt.n_hidden)
            )

        self.mu_net = nn.Linear(opt.n_hidden, opt.n_actions)
        self.logvar_net = nn.Linear(opt.n_hidden, opt.n_actions)

            
        self.cntr = 0
        self.p_z = []


    def save_z(self, z):
        if len(self.p_z) == 0:
            self.p_z = z.data.cpu()
        else:
            self.p_z = torch.cat((self.p_z, z.data.cpu()), 0)

    def sample(self):
        return Variable(random.choice(self.p_z).cuda().unsqueeze(0))

    def forward(self, input_images, input_states, target_images, target_states, sample=False, n_samples=1, save_z=False, dropout=0.0, normalize_inputs=False, normalize_outputs=False):

        if normalize_inputs:
            input_images = input_images.clone().float().div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = Variable(input_images.cuda()).unsqueeze(0)
            input_states = Variable(input_states.cuda()).unsqueeze(0)

        bsize = input_images.size(0)
        h_x = self.past_encoder(input_images, input_states)
        h_x = h_x.view(bsize, self.opt.hidden_size)
        h_x = self.proj_past(h_x)
        self.z = None

        if self.opt.actions_subsample == -1:
            h = h_x
        else:
            if self.cntr % self.opt.actions_subsample == 0:
                # sample new high-level action
                if not sample:
                    h_y = self.future_encoder(target_images.contiguous(), target_states.contiguous())
                    h_y = h_y.view(bsize, self.opt.hidden_size)
                    h_y = self.proj_future(h_y)
                    self.z = F.tanh(self.fc_z(h_x + h_y))
                    if save_z:
                        self.save_z(self.z)
                else:
                    self.z = self.sample()
            if random.random() < dropout:
                self.z = Variable(self.z.data.clone().zero_())
            h = h_x + self.z_exp(self.z)
        h = self.fc(h)
        mu = self.mu_net(h).view(bsize, self.opt.n_actions)
        logvar = self.logvar_net(h).view(bsize, self.opt.n_actions)
        logvar = torch.clamp(logvar, max = 4.0)
        std = logvar.mul(0.5).exp_()
        eps = Variable(torch.randn(bsize, n_samples, self.opt.n_actions).cuda())
        a = eps * std.view(bsize, 1, self.opt.n_actions)
        a = a + mu.view(bsize, 1, self.opt.n_actions)
        entropy = torch.mean(std)
        self.cntr += 1

        if normalize_outputs:
            a = a.data
            a.clamp_(-3, 3)
            a *= self.stats['a_std'].view(1, 1, 2).expand(a.size()).cuda()
            a += self.stats['a_mean'].view(1, 1, 2).expand(a.size()).cuda()

        return a.squeeze(), entropy, mu, std, self.z



class PolicyVAE(nn.Module):
    def __init__(self, opt):
        super(PolicyVAE, self).__init__()
        self.opt = opt
        self.past_encoder = encoder(opt, 0, opt.ncond)
        self.proj_past = nn.Linear(opt.hidden_size, opt.n_hidden)

        self.fc = nn.Sequential(
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden)
        )

        if opt.actions_subsample != -1:
            self.future_encoder = encoder(opt, 0, opt.actions_subsample)
            self.proj_future = nn.Linear(opt.hidden_size, opt.n_hidden)
            self.fc_z = nn.Sequential(
                nn.Linear(opt.n_hidden, opt.n_hidden),
                nn.ReLU(),
                nn.Linear(opt.n_hidden, opt.n_hidden),
                nn.ReLU(),
                nn.Linear(opt.n_hidden, 2*self.opt.context_dim)
            )

            self.fc_z_prior = nn.Sequential(
                nn.Linear(opt.n_hidden, opt.n_hidden),
                nn.ReLU(),
                nn.Linear(opt.n_hidden, opt.n_hidden),
                nn.ReLU(),
                nn.Linear(opt.n_hidden, 2*self.opt.context_dim)
            )


            self.z_exp = nn.Sequential(
                nn.Linear(opt.context_dim, opt.n_hidden), 
                nn.ReLU(), 
                nn.Linear(opt.n_hidden, opt.n_hidden), 
                nn.ReLU(), 
                nn.Linear(opt.n_hidden, opt.n_hidden)
            )

        self.mu_net = nn.Linear(opt.n_hidden, opt.n_actions)
        self.logvar_net = nn.Linear(opt.n_hidden, opt.n_actions)
            
        self.cntr = 0
        self.p_z = []


    def save_z(self, z):
        if len(self.p_z) == 0:
            self.p_z = z.data.cpu()
        else:
            self.p_z = torch.cat((self.p_z, z.data.cpu()), 0)

    def reparameterize(self, mu, logvar, sample):
        if self.training or sample:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


    def forward(self, input_images, input_states, target_images, target_states, sample=False, n_samples=1, save_z=False, dropout=0.0, normalize_inputs=False, normalize_outputs=False):

        if normalize_inputs:
            input_images = input_images.clone().float().div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = Variable(input_images.cuda()).unsqueeze(0)
            input_states = Variable(input_states.cuda()).unsqueeze(0)

        bsize = input_images.size(0)
        h_x = self.past_encoder(input_images, input_states)
        h_x = h_x.view(bsize, self.opt.hidden_size)
        h_x = self.proj_past(h_x)
        kld = Variable(torch.zeros(1).cuda())

        if self.opt.actions_subsample == -1:
            h = h_x
        else:
            if self.cntr % self.opt.actions_subsample == 0:
                # sample new high-level action
                mu_logvar_prior = self.fc_z_prior(h_x).view(bsize, self.opt.context_dim, 2)
                mu_prior = mu_logvar_prior[:, :, 0]
                logvar_prior = mu_logvar_prior[:, :, 1]

                if not sample:
                    h_y = self.future_encoder(target_images.contiguous(), target_states.contiguous())
                    h_y = h_y.view(bsize, self.opt.hidden_size)
                    h_y = self.proj_future(h_y)
                    mu_logvar = self.fc_z(h_x + h_y).view(bsize, self.opt.context_dim, 2)
                    mu = mu_logvar[:, :, 0]
                    logvar = mu_logvar[:, :, 1]
                    self.z = self.reparameterize(mu, logvar, sample=True)
                    kld = utils.kl_criterion(mu, logvar, mu_prior, logvar_prior)
                    if save_z:
                        self.save_z(self.z)
                else:
                    self.z = self.reparamaterize(mu_prior, logvar_prior, sample=True)
            if random.random() < dropout:
                self.z = Variable(self.z.data.clone().zero_())
            h = h_x + self.z_exp(self.z)
        h = self.fc(h)
        mu = self.mu_net(h).view(bsize, self.opt.n_actions)
        logvar = self.logvar_net(h).view(bsize, self.opt.n_actions)
        logvar = torch.clamp(logvar, max = 4.0)
        std = logvar.mul(0.5).exp_()
        eps = Variable(torch.randn(bsize, n_samples, self.opt.n_actions).cuda())
        a = eps * std.view(bsize, 1, self.opt.n_actions)
        a = a + mu.view(bsize, 1, self.opt.n_actions)
        entropy = torch.mean(std)
        self.cntr += 1

        if normalize_outputs:
            a = a.data
            a.clamp_(-3, 3)
            a *= self.stats['a_std'].view(1, 1, 2).expand(a.size()).cuda()
            a += self.stats['a_mean'].view(1, 1, 2).expand(a.size()).cuda()

        return a.squeeze(), kld, mu, std


        
        




class DeterministicPolicy(nn.Module):
    def __init__(self, opt, context_dim=0, output_dim=None):
        super(DeterministicPolicy, self).__init__()
        self.opt = opt
        self.encoder = encoder(opt, 0, opt.ncond)
        self.hsize = opt.nfeature*self.opt.h_height*self.opt.h_width
        self.n_outputs = opt.n_actions if output_dim is None else output_dim
        self.proj = nn.Linear(self.hsize, opt.n_hidden)
        self.context_dim = context_dim

        self.fc = nn.Sequential(
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, self.n_outputs)
        )

        if context_dim > 0:
            self.context_encoder = nn.Sequential(
                nn.Linear(context_dim, opt.n_hidden), 
                nn.ReLU(), 
                nn.Linear(opt.n_hidden, opt.n_hidden), 
                nn.ReLU(), 
                nn.Linear(opt.n_hidden, opt.n_hidden)
            )



    def forward(self, state_images, states, context=None, sample=True, normalize_inputs=False, normalize_outputs=False, n_samples=1):
        bsize = state_images.size(0)
        h = self.encoder(state_images, states).view(bsize, self.hsize)
        h = self.proj(h)
        if self.context_dim > 0:
            assert(context is not None)
            h = h + self.context_encoder(context)
        h = self.fc(h).view(bsize, self.n_outputs)
        return h


    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()



# Mixture Density Network model
class PolicyMDN(nn.Module):
    def __init__(self, opt, n_mixture=10, npred=1):
        super(PolicyMDN, self).__init__()
        self.opt = opt
        self.npred = npred
        if not hasattr(opt, 'n_mixture'):
            self.opt.n_mixture = n_mixture
        self.encoder = encoder(opt, 0, opt.ncond)
        self.hsize = opt.nfeature*self.opt.h_height*self.opt.h_width
        self.n_outputs = self.npred*opt.n_actions
        self.fc = nn.Sequential(
            nn.Linear(self.hsize, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden)
        )

        self.pi_net = nn.Linear(opt.n_hidden, opt.n_mixture)
        self.mu_net = nn.Linear(opt.n_hidden, opt.n_mixture*self.n_outputs)
        self.sigma_net = nn.Linear(opt.n_hidden, opt.n_mixture*self.n_outputs)


    def forward(self, state_images, states, sample=False, unnormalize=False):

        if unnormalize:
            # policy network is trained with states normalized by mean and standard dev. 
            # this is to unnormalize the predictions at evaluation time. 
            state_images = state_images.clone().float().div_(255.0)
            states -= self.stats['s_mean'].view(1, 4).expand(states.size())
            states /= self.stats['s_std'].view(1, 4).expand(states.size())
            state_images = Variable(state_images.cuda()).unsqueeze(0)
            states = Variable(states.cuda()).unsqueeze(0)


        bsize = state_images.size(0)
        h = self.encoder(state_images, states).view(bsize, self.hsize)
        h = self.fc(h)
        # get parameters of output distribution
        pi = F.softmax(self.pi_net(h).view(bsize, self.opt.n_mixture), dim=1)
        mu = self.mu_net(h).view(bsize, self.opt.n_mixture, self.n_outputs)
        sigma = F.softplus(self.sigma_net(h)).view(bsize, self.opt.n_mixture, self.n_outputs)
        if sample:
            # pick a mixture component (one for each element in minibatch)
            k = torch.multinomial(pi, 1)
            a = []
            for b in range(bsize):
                # sample from Gaussian associated with those components
                a.append(torch.randn(self.npred, self.opt.n_actions).cuda()*sigma[b][k[b]].data + mu[b][k[b]].data)
            a = torch.stack(a).squeeze()
            a = a.view(bsize, self.npred, 2)
        else:
            a = None

        if unnormalize:
            a *= self.stats['a_std'].view(1, 1, 2).expand(a.size()).cuda()
            a += self.stats['a_mean'].view(1, 1, 2).expand(a.size()).cuda()

        return pi, mu, sigma, a


    def intype(self, t):
        if t == 'gpu':
            self.cuda()
            if hasattr(self, 'a_mean'):
                self.a_mean = self.a_mean.cuda()
                self.a_std = self.a_std.cuda()
                self.s_mean = self.s_mean.cuda()
                self.s_std = self.s_std.cuda()
        elif t == 'cpu':
            self.cpu()
            if hasattr(self, 'a_mean'):
                self.a_mean = self.a_mean.cpu()
                self.a_std = self.a_std.cpu()
                self.s_mean = self.s_mean.cpu()
                self.s_std = self.s_std.cpu()
