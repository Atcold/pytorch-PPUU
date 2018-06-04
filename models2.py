import torch
import torch.nn as nn
import torch.legacy.nn as lnn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import random, pdb, copy, os, math, numpy, copy, time
import utils




# this file will implement new changes such as fully-connected layers, LSTMs etc
# we keep old models.py for now, for backward compatibility



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
        self.f_encoder = nn.Sequential(
            nn.Conv2d(3*self.n_inputs, opt.nfeature, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1),
        )

        if states:
            # state encoder
            self.s_encoder = nn.Sequential(
                nn.Linear(state_input_size*opt.ncond, opt.nfeature),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(opt.nfeature, opt.nfeature),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(opt.nfeature, self.opt.hidden_size)
            )

        if a_size > 0:
            # action or cost encoder
            self.a_encoder = nn.Sequential(
                nn.Linear(a_size, opt.nfeature),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(opt.nfeature, opt.nfeature),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(opt.nfeature, self.opt.hidden_size)
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




# decodes a hidden state into a predicted frame, a predicted state and a predicted cost vector
class decoder(nn.Module):
    def __init__(self, opt, n_out=1):
        super(decoder, self).__init__()
        self.opt = opt
        self.n_out = n_out
        # minor adjustments to make output size same as input
        if self.opt.dataset == 'simulated':
            self.f_decoder = nn.Sequential(
                nn.ConvTranspose2d(opt.nfeature, opt.nfeature, (4, 5), 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(opt.nfeature, opt.nfeature, (5, 5), 2, (1, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(opt.nfeature, 3, (2, 2), 2, (0, 1))
            )
        elif self.opt.dataset == 'i80':
            self.f_decoder = nn.Sequential(
                nn.ConvTranspose2d(opt.nfeature, opt.nfeature, (4, 4), 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(opt.nfeature, opt.nfeature, (5, 5), 2, (0, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(opt.nfeature, self.n_out*3, (2, 2), 2, (0, 1))
            )

            self.h_reducer = nn.Sequential(
                nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(opt.nfeature, opt.nfeature, (4, 1), (2, 1), 0),
                nn.LeakyReLU(0.2, inplace=True)
            )

            self.c_predictor = nn.Sequential(
                nn.Linear(2*opt.nfeature, opt.nfeature),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(opt.nfeature, opt.nfeature),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(opt.nfeature, self.n_out*2),
                nn.Sigmoid()
            )

            self.s_predictor = nn.Sequential(
                nn.Linear(2*opt.nfeature, opt.nfeature),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(opt.nfeature, opt.nfeature),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(opt.nfeature, self.n_out*4)
            )


    def forward(self, h):
        bsize = h.size(0)
        h = h.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
        h_reduced = self.h_reducer(h).view(bsize, -1)
        pred_cost = self.c_predictor(h_reduced)
        pred_state = self.s_predictor(h_reduced)
        pred_image = self.f_decoder(h)[:, :, :-1].clone().view(bsize, 1, 3*self.n_out, self.opt.height, self.opt.width)
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



# encodes a sequence of frames and states, and produces a distribution over latent variables. 
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













# Mixture Density network (fully-connected). 
class u_network_mdn_fc(nn.Module):
    def __init__(self, opt, n_outputs):
        super(u_network_mdn_fc, self).__init__()
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






# forward TEN model
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

        self.y_encoder = encoder(opt, opt.n_actions, opt.ncond + 1)

        self.z_network = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nz)
        )

        # if beta > 0, it means we are training a prior network jointly
        if self.opt.beta > 0:
            self.u_network = u_network_mdn_fc(opt, opt.nz)

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
            if hasattr(self, 'u_network'):
                # this is if the likelihood network has been trained jointly
                h_x = self.encoder(input_images, input_states, action)
                pi, mu, sigma = self.u_network(h_x)
            elif hasattr(self, 'prior'):
                # this is if the likelihood network has been trained separately
                pi, mu, sigma = self.prior(input_images, input_states, action)
            bsize = mu.size(0)
            tt = time.time()
            # sample a z vector from the likelihood model
            pi_sample = torch.multinomial(pi, 1).squeeze()
            mu_sample = torch.gather(mu, dim=1, index=pi_sample.view(bsize, 1, 1).expand(bsize, 1, nz)).squeeze()
            sigma_sample = torch.gather(sigma, dim=1, index=pi_sample.view(bsize, 1, 1).expand(bsize, 1, nz)).squeeze()
            z_sample = mu_sample.data + torch.randn(bsize, nz).cuda()*sigma_sample.data
            # quantize it to its nearest neighbor
            dist = torch.norm(self.p_z.view(-1, 1, nz) - z_sample.view(1, -1, nz), 2, 2)
            _, z_ind = torch.topk(dist, 1, dim=0, largest=False)
            z = self.p_z.index(z_ind.squeeze().cuda())
            tt = time.time() - tt
#            print(f'NN search took {tt}s')

        if self.use_cuda: z = z.cuda()
        return [Variable(z), z_indx]

    def forward(self, inputs, actions, targets, save_z = False, sampling=None, p_dropout=0.0):
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
                h_y = self.y_encoder(torch.cat((input_images, target_images[:, t].unsqueeze(1).contiguous()), 1), input_states, actions[:, t])
                if random.random() < p_dropout:
                    z = Variable(torch.zeros(bsize, self.opt.nz).cuda())
                else:
                    z = self.z_network(utils.combine(h_x, h_y, self.opt.combine).view(bsize, -1))
                    if save_z:
                        self.save_z(z)
                    if self.opt.beta > 0:
                        pi, mu, sigma = self.u_network(h_x)
                        ploss = utils.mdn_loss_fn(pi, sigma, mu, z)
                        if math.isnan(ploss.data[0]):
                            pdb.set_trace()
            else:
                # we are doing inference
                z = self.sample_z(bsize, sampling, input_images, input_states, actions[:, t], z, t0=False)

            z_ = z if sampling is None else z[0]
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)

            pred_image, pred_state, pred_cost = self.decoder(h)
            if sampling is not None:
                pred_image.detach()
                pred_state.detach()
                pred_cost.detach()
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
        return [pred_images, pred_states, pred_costs], ploss


    def plan_actions_backprop(self, observation, args):
        input_images, input_states = observation
        input_images = Variable(input_images.cuda().float()).unsqueeze(0)
        input_states = Variable(input_states.cuda()).unsqueeze(0)
        bsize = input_images.size(0)
        # repeat for multiple rollouts
        actions = Variable(torch.randn(bsize, args.npred, self.opt.n_actions).cuda(), requires_grad=True)
        optimizer_a = optim.Adam([actions], args.lrt)
        for i in range(0, args.n_iter):
            optimizer_a.zero_grad()
            pred, _ = self.forward([input_images, input_states], actions, None, sampling='fp')
            pdb.set_trace()
            costs = pred[2]
            loss = costs.mean()
            loss.backward()
            if verbose:
                print('[iter {} | mean cost = {}], grad = {}'.format(i, loss.data[0], 0))
            optimizer_a.step()

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
        actions = Variable(torch.zeros(bsize, args.npred, self.opt.n_actions).cuda().mul_(0.01), requires_grad=True)
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


# Mixture Density Network model
class PolicyMDN(nn.Module):
    def __init__(self, opt):
        super(PolicyMDN, self).__init__()
        self.opt = opt
        self.encoder = encoder(opt, 0, opt.ncond)
        self.hsize = opt.nfeature*self.opt.h_height*self.opt.h_width
        self.n_outputs = opt.npred*opt.n_actions
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
        bsize = state_images.size(0)
        h = self.encoder(state_images, states).view(bsize, self.hsize)
        h = self.fc(h)
        pi = F.softmax(self.pi_net(h).view(bsize, self.opt.n_mixture), dim=1)
        mu = self.mu_net(h).view(bsize, self.opt.n_mixture, self.n_outputs)
        sigma = F.softplus(self.sigma_net(h)).view(bsize, self.opt.n_mixture, self.n_outputs)
        if sample:
            k = torch.multinomial(pi, 1)
            a = []
            for b in range(bsize):
                a.append(torch.randn(self.opt.npred, self.opt.n_actions)*sigma[b][k[b]].data + mu[b][k[b]].data)
            a = torch.stack(a).squeeze()
            a[:, 1].copy_(torch.clamp(a[:, 1], min=-1, max=1))
            print(print('a:{}, {}'.format(a.min(), a.max())))
        else:
            a = None

        if unnormalize:
            a *= self.a_std
            a += self.a_mean


        return pi, mu, sigma, a


    def intype(self, t):
        if t == 'gpu':
            self.cuda()
            if self.a_mean is not None:
                self.a_mean = self.a_mean.cuda()
                self.a_std = self.a_std.cuda()
                self.s_mean = self.s_mean.cuda()
                self.s_std = self.s_std.cuda()
        elif t == 'cpu':
            self.cpu()
            if self.a_mean is not None:
                self.a_mean = self.a_mean.cpu()
                self.a_std = self.a_std.cpu()
                self.s_mean = self.s_mean.cpu()
                self.s_std = self.s_std.cpu()
