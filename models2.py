import torch
import torch.nn as nn
import torch.legacy.nn as lnn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import random, pdb, copy, os, math, numpy
import utils




# this file will implement new changes such as fully-connected layers, LSTMs etc
# we keep old models.py for now, for backward compatibility



####################
# Basic modules
####################

# encodes a sequence of input frames and states, and optionally a cost or action, to a hidden representation
class encoder(nn.Module):
    def __init__(self, opt, a_size, n_inputs, states=True):
        super(encoder, self).__init__()
        self.opt = opt
        self.a_size = a_size
        self.n_inputs = opt.ncond if n_inputs is None else n_inputs
        # frame encoder
        self.f_encoder = nn.Sequential(
            nn.Conv2d(3*self.n_inputs, opt.nfeature, 4, 2, 1),
            #nn.BatchNorm2d(opt.nfeature),
            nn.LeakyReLU(0.2),
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1),
            #nn.BatchNorm2d(opt.nfeature),
            nn.LeakyReLU(0.2),
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1),
            #nn.BatchNorm2d(opt.nfeature)
        )

        if states:
            # state encoder
            self.s_encoder = nn.Sequential(
                #nn.BatchNorm1d(4*self.n_inputs),
                nn.Linear(4*self.n_inputs, opt.nfeature),
                #nn.BatchNorm1d(opt.nfeature),
                nn.LeakyReLU(0.2),
                nn.Linear(opt.nfeature, opt.nfeature),
                #nn.BatchNorm1d(opt.nfeature),
                nn.LeakyReLU(0.2),
                nn.Linear(opt.nfeature, self.opt.hidden_size)
            )

        if a_size > 0:
            # action or cost encoder
            self.a_encoder = nn.Sequential(
                #nn.BatchNorm1d(a_size),
                nn.Linear(a_size, opt.nfeature),
                #nn.BatchNorm1d(opt.nfeature),
                nn.LeakyReLU(0.2),
                nn.Linear(opt.nfeature, opt.nfeature),
                #nn.BatchNorm1d(opt.nfeature),
                nn.LeakyReLU(0.2),
                nn.Linear(opt.nfeature, self.opt.hidden_size)
            )

    def forward(self, images, states=None, actions=None):
        bsize = images.size(0)
        h = self.f_encoder(images.view(bsize, self.n_inputs*3, self.opt.height, self.opt.width))
        if states is not None:
            h = utils.combine(h, self.s_encoder(states.contiguous().view(bsize, self.n_inputs*4)).view(h.size()), self.opt.combine)
        if actions is not None:
            a = self.a_encoder(actions.contiguous().view(bsize, self.a_size))
            h = utils.combine(h, a.view(h.size()), self.opt.combine)
        return h


# decodes a hidden state into a predicted frame and a set of predicted costs
class decoder(nn.Module):
    def __init__(self, opt, n_out=1):
        super(decoder, self).__init__()
        self.opt = opt
        self.n_out = n_out
        # minor adjustments to make output size same as input
        if self.opt.dataset == 'simulated':
            self.f_decoder = nn.Sequential(
                nn.ConvTranspose2d(opt.nfeature, opt.nfeature, (4, 5), 2, 1),
                #nn.BatchNorm2d(opt.nfeature),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(opt.nfeature, opt.nfeature, (5, 5), 2, (1, 1)),
                #nn.BatchNorm2d(opt.nfeature),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(opt.nfeature, 3, (2, 2), 2, (0, 1))
            )
        elif self.opt.dataset == 'i80':
            self.f_decoder = nn.Sequential(
                nn.ConvTranspose2d(opt.nfeature, opt.nfeature, (4, 4), 2, 1),
                #nn.BatchNorm2d(opt.nfeature),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(opt.nfeature, opt.nfeature, (5, 5), 2, (0, 1)),
                #nn.BatchNorm2d(opt.nfeature),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(opt.nfeature, self.n_out*3, (2, 2), 2, (0, 1))
            )

            self.h_reducer = nn.Sequential(
                nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1),
                #nn.BatchNorm2d(opt.nfeature),
                nn.LeakyReLU(0.2),
                nn.Conv2d(opt.nfeature, opt.nfeature, (4, 1), (2, 1), 0),
                #nn.BatchNorm2d(opt.nfeature),
                nn.LeakyReLU(0.2)
            )

            self.c_predictor = nn.Sequential(
                nn.Linear(2*opt.nfeature, opt.nfeature),
                #nn.BatchNorm1d(opt.nfeature),
                nn.LeakyReLU(0.2),
                nn.Linear(opt.nfeature, opt.nfeature),
                #nn.BatchNorm1d(opt.nfeature),
                nn.LeakyReLU(0.2),
                nn.Linear(opt.nfeature, self.n_out*2),
                nn.Sigmoid()
            )

            self.s_predictor = nn.Sequential(
                nn.Linear(2*opt.nfeature, opt.nfeature),
                #nn.BatchNorm1d(opt.nfeature),
                nn.LeakyReLU(0.2),
                nn.Linear(opt.nfeature, opt.nfeature),
                #nn.BatchNorm1d(opt.nfeature),
                nn.LeakyReLU(0.2),
                nn.Linear(opt.nfeature, self.n_out*4)
            )

            if self.opt.decoder > 0:
                self.u_encoder = nn.Sequential(
                    nn.Conv2d(opt.nfeature, 2*opt.nfeature, 4, 2, 1),
                    #nn.BatchNorm2d(2*opt.nfeature),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(2*opt.nfeature, 4*opt.nfeature, (4, 1), 2, (1, 0)),
                    #nn.BatchNorm2d(4*opt.nfeature),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(4*opt.nfeature, 8*opt.nfeature, (3, 1), 2, (0, 0)),
                    #nn.BatchNorm2d(8*opt.nfeature)
                )
                self.u_decoder = nn.Sequential(
                    nn.ConvTranspose2d(8*opt.nfeature, 4*opt.nfeature, (3, 2), 2, 0),
                    #nn.BatchNorm2d(4*opt.nfeature),
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose2d(4*opt.nfeature, 2*opt.nfeature, (4, 1), 2, (1, 0)),
                    #nn.BatchNorm2d(2*opt.nfeature),
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose2d(2*opt.nfeature, opt.nfeature, (4, 1), 2, (0, 1)),
                    #nn.BatchNorm2d(opt.nfeature)
                )


    def forward(self, h):
        bsize = h.size(0)
        if self.opt.decoder == 0:
            h = h
        elif self.opt.decoder == 1:
            h = self.u_decoder(self.u_encoder(h))
        elif self.opt.decoder == 2:
            h = h + self.u_decoder(self.u_encoder(h))
        h = h.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
        h_reduced = self.h_reducer(h).view(bsize, -1)
        pred_cost = self.c_predictor(h_reduced)
        pred_state = self.s_predictor(h_reduced)
        pred_image = self.f_decoder(h)[:, :, :-1].clone().view(bsize, 1, 3*self.n_out, self.opt.height, self.opt.width)
        return pred_image, pred_state, pred_cost




# encodes a sequence of frames or errors and produces a distribution over latent variables
class z_network_gaussian(nn.Module):
    def __init__(self, opt):
        super(z_network_gaussian, self).__init__()
        self.opt = opt

        self.network = nn.Sequential(
            nn.Linear(opt.nfeature*self.opt.h_height*self.opt.h_width, opt.nfeature),
            nn.LeakyReLU(0.2),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2),
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

    def forward(self, inputs, sample=False):
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar, sample)
        return z, mu, logvar





# takes as input a sequence of frames, outputs the means and variances of a diagonal Gaussian.
class u_network_gaussian(nn.Module):
    def __init__(self, opt, n_inputs):
        super(u_network_gaussian, self).__init__()
        self.opt = opt
        self.n_inputs = n_inputs

        self.conv = nn.Sequential(
            nn.Conv2d(3*n_inputs, opt.nfeature, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1),
        )

        self.fc = nn.Sequential(
            nn.Linear(opt.nfeature*self.opt.h_height*self.opt.h_width, opt.nfeature),
            nn.LeakyReLU(0.2),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2),
            nn.Linear(opt.nfeature, 2*opt.nz)
        )

    def forward(self, inputs):
        bsize = inputs.size(0)
        inputs = inputs.view(bsize, self.n_inputs*3, self.opt.height, self.opt.width)
        z = self.fc(self.conv(inputs).view(bsize, -1))
        mu = z[:, :self.opt.nz]
        sigma = F.softplus(z[:, self.opt.nz:] )
        return mu, sigma








# Mixture Density network (fully-connected)
class u_network_mdn_fc(nn.Module):
    def __init__(self, opt, n_outputs):
        super(u_network_mdn_fc, self).__init__()
        self.opt = opt
        self.n_outputs = n_outputs

        self.network = nn.Sequential(
            nn.Linear(self.opt.hidden_size, opt.nfeature),
            nn.LeakyReLU(0.2),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2)
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










# expands a latent variable to the size of the hidden representation
class z_expander(nn.Module):
    def __init__(self, opt, n_steps):
        super(z_expander, self).__init__()
        self.opt = opt
        self.n_steps = n_steps
        self.z_expander = nn.Sequential(
            nn.Linear(opt.nz, opt.nfeature),
            nn.LeakyReLU(0.2),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2),
            nn.Linear(opt.nfeature, n_steps * opt.nfeature * self.opt.h_height * self.opt.h_width)
        )

    def forward(self, z):
        bsize = z.size(0)
        z_exp = self.z_expander(z).view(bsize, self.n_steps, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
        return z_exp



# combines a sequence of images with the state representing absolute speed
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
        if mfile == '':
            self.encoder = encoder(opt, opt.n_actions, opt.ncond)
            self.decoder = decoder(opt)
        else:
            print(f'[initializing encoder and decoder with: {mfile}]')
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




# forward model, Mixture Density Network.
class FwdCNN_MDN(nn.Module):
    def __init__(self, opt, mfile):
        super(FwdCNN_MDN, self).__init__()
        self.opt = opt
        self.encoder = encoder(opt, opt.n_actions, opt.ncond)
        self.decoder = decoder(opt, n_out = 2*opt.n_mixture)

        self.pi_network = nn.Sequential(
            nn.Linear(self.opt.hidden_size, opt.nfeature),
            nn.LeakyReLU(0.2),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2),
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

#            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
#            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
#            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
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






# forward AE model with a fixed prior
class FwdCNN_AE_FP(nn.Module):
    def __init__(self, opt, mfile=''):
        super(FwdCNN_AE_FP, self).__init__()
        self.opt = opt

        if mfile == '':
            self.encoder = encoder(opt, opt.n_actions, opt.ncond)
            self.decoder = decoder(opt)
        else:
            print(f'[initializing encoder and decoder with: {mfile}]')
            self.mfile = mfile
            pretrained_model = torch.load(mfile)
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.encoder.n_inputs = opt.ncond
            self.decoder.n_out = 1

        self.y_encoder = encoder(opt, 0, 1, states=False)

        self.z_network = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.nfeature),
            nn.LeakyReLU(0.2),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2),
            nn.Linear(opt.nfeature, opt.nz)
        )

        self.u_network = u_network_mdn_fc(opt, opt.nz)

        self.z_expander = nn.Linear(opt.nz, opt.hidden_size)
        self.p_z = []

    def save_z(self, z):
        if len(self.p_z) == 0:
            self.p_z = z.data.cpu()
        else:
            self.p_z = torch.cat((self.p_z, z.data.cpu()), 0)

    def sample_z(self, bsize, method='fp', input_images=None, input_states=None, action=None):
        z = []
        if method == 'fp':
            for b in range(bsize):
                z.append(random.choice(self.p_z))
            z = torch.stack(z).contiguous()
        elif method == 'dist':
            h_x = self.encoder(input_images, input_states, action)
            pi, mu, sigma = self.u_network(h_x)
            k = torch.multinomial(pi, 1)
            for b in range(bsize):
                z.append(torch.randn(self.opt.nz).cuda()*sigma[b][k[b]].data + mu[b][k[b]].data)
            z = torch.stack(z).squeeze()

        elif method == 'pdf':
            n_z = bsize*1000
            if len(self.Z) == 0:
                self.Z = self.sample_z(n_z)

            h_x = self.encoder(input_images, input_states, action)
            pi, mu, sigma = self.u_network(h_x)
            mu = mu.contiguous()
            sigma = sigma.contiguous()
            Z_exp = self.Z.view(1, n_z, self.opt.nz).expand(bsize, n_z, self.opt.nz)
            mu_exp = mu.view(bsize, 1, self.opt.n_mixture, self.opt.nz).expand(bsize, n_z, self.opt.n_mixture, self.opt.nz)
            sigma_exp = sigma.view(bsize, 1, self.opt.n_mixture, self.opt.nz).expand(bsize, n_z, self.opt.n_mixture, self.opt.nz)
            mu_exp = mu_exp.contiguous().view(-1, self.opt.n_mixture, self.opt.nz)
            sigma_exp = sigma_exp.contiguous().view(-1, self.opt.n_mixture, self.opt.nz)
            pi_exp = pi.view(bsize, 1, self.opt.n_mixture).expand(bsize, n_z, self.opt.n_mixture).contiguous().view(-1, self.opt.n_mixture)
            Z_exp = Z_exp.clone().view(-1, self.opt.nz)
            z_loss = utils.mdn_loss_fn(pi_exp, sigma_exp, mu_exp, Z_exp, avg=False)
            z_loss = z_loss.view(bsize, -1)
            _, z_ind = torch.topk(z_loss, self.opt.topz_sample, dim=1, largest=False)
            z_top = Variable(self.Z.data.index(z_ind.data.view(-1))).view(bsize, self.opt.topz_sample, self.opt.nz)
            z_ind = random.choice(z_ind.t())
            z = self.Z.data.index(z_ind.data)
            self.z_top_list.append(z_top.data)

            if random.random() < 0.01 and False:
                # save
                print('[saving]')
                hsh = random.randint(0, 10000)
                Z = Z.data.cpu().numpy()
                z_top = z_top.data.cpu().numpy()
                embed = utils.embed(Z, z_top)
                os.system('mkdir -p z_viz/pdf_{}samples'.format(n_sample))
                torch.save(embed, 'z_viz/pdf_{}samples/{:05d}.pth'.format(n_sample, hsh))
            del mu, sigma, Z_exp, mu_exp, sigma_exp


        if self.use_cuda: z = z.cuda()
        return Variable(z)

    def forward(self, inputs, actions, targets, save_z = False, sampling='fp'):
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

        for t in range(npred):
            h_x = self.encoder(input_images, input_states, actions[:, t])
            if targets is not None:
                target_images, target_states, target_costs = targets
                # we are training or estimating z distribution
                h_y = self.y_encoder(target_images[:, t].unsqueeze(1).contiguous())
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
                z = self.sample_z(bsize, sampling, input_images, input_states, actions[:, t])



            z_exp = self.z_expander(z).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
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
        return [pred_images, pred_states, pred_costs], ploss

    def intype(self, t):
        if t == 'gpu':
            self.cuda()
            self.use_cuda = True
        elif t == 'cpu':
            self.cpu()
            self.use_cuda = False





# forward VAE model with a learned prior
class FwdCNN_VAE_FP(nn.Module):
    def __init__(self, opt, mfile=''):
        super(FwdCNN_VAE_FP, self).__init__()
        self.opt = opt
        if mfile == '':
            self.encoder = encoder(opt, opt.n_actions, opt.ncond)
            self.decoder = decoder(opt)
        else:
            print(f'[initializing encoder and decoder with: {mfile}]')
            self.mfile = mfile
            pretrained_model = torch.load(mfile)
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.encoder.n_inputs = opt.ncond


        self.y_encoder = encoder(opt, 0, 1)
        self.z_network = z_network(opt)
        self.z_expander = z_expander(opt, 1)

    def forward(self, inputs, actions, targets, sampling=None):
        bsize = inputs.size(0)
        inputs = inputs.view(bsize, self.opt.ncond, 3, self.opt.height, self.opt.width)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)

        kld = Variable(torch.zeros(1))
        if self.use_cuda:
            kld = kld.cuda()
        pred = []
        for t in range(npred):
            h_x = self.encoder(inputs, actions[:, t])
            if targets is not None:
                # we are training
                h_y = self.y_encoder(targets[:, t].unsqueeze(1).contiguous())
                z, mu, logvar = self.z_network(h_x + h_y)
                logvar = torch.clamp(logvar, max = 4) # this can go to inf when taking exp(), so clamp it
                kld_t = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kld_t /= bsize
                kld += kld_t
                z_exp = self.z_expander(z)
            else:
                # we are generating samples
                z = Variable(torch.randn(bsize, self.opt.nz).cuda())
                z_exp = self.z_expander(z)

            h = h_x + z_exp.squeeze()
            pred_ = F.sigmoid(self.decoder(h) + inputs[:, -1].unsqueeze(1).clone())
            pred.append(pred_)
            inputs = torch.cat((inputs[:, 1:], pred_), 1)

        kld /= npred
        pred = torch.cat(pred, 1)
        return pred, kld

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
            print(f'[initializing encoder and decoder with: {mfile}]')
            self.mfile = mfile
            pretrained_model = torch.load(mfile)
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.encoder.n_inputs = opt.ncond


        self.y_encoder = encoder(opt, 0, 1)
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
            if targets is not None:
                # we are training
                target_images, target_states, target_costs = targets
                h_y = self.y_encoder(target_images[:, t].unsqueeze(1).contiguous())
                z1, mu1, logvar1 = self.z_network(h_x + h_y)
                z2, mu2, logvar2 = self.q_network(h_x)
                sigma1 = logvar1.mul(0.5).exp()
                sigma2 = logvar2.mul(0.5).exp()
                kld_t = torch.log(sigma2/sigma1 + 1e-6) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(1e-6 + 2*torch.exp(logvar2)) - 1/2
                kld_t = torch.clamp(kld_t, max=50)
                kld_t = torch.sum(kld_t) / bsize
                kld += kld_t
                z_exp = self.z_expander(z1)
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








# forward VAE model with a learned prior
class FwdCNN_VAE_LP1(nn.Module):
    def __init__(self, opt, mfile=''):
        super(FwdCNN_VAE_LP1, self).__init__()
        self.opt = opt
        if mfile == '':
            self.encoder = encoder(opt, opt.n_actions, opt.ncond)
            self.decoder = decoder(opt)
        else:
            print(f'[initializing encoder and decoder with: {mfile}]')
            self.mfile = mfile
            pretrained_model = torch.load(mfile)
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.encoder.n_inputs = opt.ncond


        self.y_encoder = encoder(opt, 0, 1)
        self.z_network = z_network(opt)
        self.q_network = z_network(opt)
        self.z_expander = z_expander(opt, 1)

    def forward(self, inputs, actions, targets, sampling=None):
        bsize = inputs.size(0)
        inputs = inputs.view(bsize, self.opt.ncond, 3, self.opt.height, self.opt.width)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)

        kld = Variable(torch.zeros(1))
        if self.use_cuda:
            kld = kld.cuda()
        pred = []
        for t in range(npred):
            h_x = self.encoder(inputs, actions[:, t])
            if targets is not None:
                # we are training
                h_y = self.y_encoder(targets[:, t].unsqueeze(1).contiguous())
                z1, mu1, logvar1 = self.z_network(h_x + h_y)
                z2, mu2, logvar2 = self.q_network(h_x)
                sigma1 = logvar1.mul(0.5).exp()
                sigma2 = logvar2.mul(0.5).exp()
                kld_t = torch.log(sigma2/sigma1 + 1e-8) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
                kld_t = torch.clamp(kld_t, max=50)
                kld_t = torch.sum(kld_t) / bsize
                kld += kld_t
                z_exp = self.z_expander(z1)
            else:
                # we are generating samples
                z, _, _ = self.q_network(h_x, sample=True)
                z_exp = self.z_expander(z)

            h = h_x + z_exp.squeeze()
            pred_ = F.sigmoid(self.decoder(h) + inputs[:, -1].unsqueeze(1).clone())
            pred.append(pred_)
            inputs = torch.cat((inputs[:, 1:], pred_), 1)

        kld /= npred
        pred = torch.cat(pred, 1)
        return pred, kld

    def intype(self, t):
        if t == 'gpu':
            self.cuda()
            self.use_cuda = True
        elif t == 'cpu':
            self.cpu()
            self.use_cuda = False







class LSTMCritic(nn.Module):
    def __init__(self, opt):
        super(LSTMCritic, self).__init__()
        self.opt = opt
        self.encoder = encoder(opt, 0, 1)
        self.input_size = opt.nfeature*opt.h_height*opt.h_width
        self.lstm = nn.LSTM(self.input_size, opt.nhidden)
        self.classifier = nn.Linear(opt.nhidden, 1)
        self.hidden = (Variable(torch.randn(1, 2*opt.batch_size, opt.nhidden).cuda()),
                       Variable(torch.randn(1, 2*opt.batch_size, opt.nhidden).cuda()))


    def forward(self, inputs):
        self.hidden = (Variable(torch.randn(1, 2*self.opt.batch_size, self.opt.nhidden).cuda()),
                       Variable(torch.randn(1, 2*self.opt.batch_size, self.opt.nhidden).cuda()))
        inputs.detach()
        bsize = inputs.size(0)
        T = inputs.size(1)
        for t in range(T):
            enc = self.encoder(inputs[:, t].contiguous()).view(1, 2*self.opt.batch_size, -1)
            out, self.hidden = self.lstm(enc, self.hidden)
        logits = self.classifier(out.squeeze())
        return logits.squeeze()







class PolicyCNN(nn.Module):
    def __init__(self, opt):
        super(PolicyCNN, self).__init__()
        self.opt = opt
        self.encoder = encoder(opt)
#        self.encoder = policy_encoder(opt)
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
        print(pi)
        if sample:
            k = torch.multinomial(pi, 1)
            a = []
            for b in range(bsize):
                a.append(torch.randn(self.opt.n_actions)*sigma[b][k[b]].data + mu[b][k[b]].data)
            a = torch.stack(a).squeeze()
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













class PolicyAE(nn.Module):
    def __init__(self, opt, mfile):
        super(PolicyAE, self).__init__()
        self.opt = opt
        self.hsize = opt.nfeature*self.opt.h_height*self.opt.h_width
        self.p_z = []

        if mfile == '':
            self.encoder = policy_encoder(opt)
            self.fc = nn.Sequential(
                nn.Linear(self.hsize + opt.n_hidden, opt.n_hidden),
#                nn.BatchNorm1d(opt.n_hidden),
                nn.ReLU(),
                nn.Linear(opt.n_hidden, opt.n_hidden),
#                nn.BatchNorm1d(opt.n_hidden),
                nn.ReLU(),
                nn.Linear(opt.n_hidden, opt.npred*opt.n_actions)
            )

            self.a_encoder = nn.Sequential(
                nn.Linear(self.opt.npred*self.opt.n_actions, opt.n_hidden),
#                nn.BatchNorm1d(opt.n_hidden),
                nn.ReLU(),
                nn.Linear(opt.n_hidden, opt.n_hidden),
#                nn.BatchNorm1d(opt.n_hidden),
                nn.ReLU(),
                nn.Linear(opt.n_hidden, opt.n_hidden)
            )

        else:
            print(f'[initializing encoder and decoder with: {mfile}]')
            pretrained_model = torch.load(mfile)
            self.encoder1 = pretrained_model.encoder
            self.fc1 = pretrained_model.fc
            self.encoder2 = copy.deepcopy(pretrained_model.encoder)
            self.fc2 = copy.deepcopy(pretrained_model.fc)


        self.z_network = nn.Sequential(
            nn.BatchNorm1d(self.hsize + 2*opt.n_hidden),
            nn.Linear(self.hsize + 2*opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(opt.n_hidden),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(opt.n_hidden),
            nn.Linear(opt.n_hidden, opt.nz)
            )

        self.z_expander = nn.Sequential(
            nn.Linear(opt.nz, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, self.hsize + opt.n_hidden)
        )


    def save_z(self, z):
        if len(self.p_z) == 0:
            self.p_z = z.data.cpu()
        else:
            self.p_z = torch.cat((self.p_z, z.data.cpu()), 0)

    def sample_z(self, bsize):
        z = []
        for b in range(bsize):
            z.append(random.choice(self.p_z))
        z = torch.stack(z)
        if self.use_cuda: z = z.cuda()
        return Variable(z)


    def forward(self, state_images, states, actions, save_z=False):
        bsize = state_images.size(0)
        h = self.encoder(state_images, states)
        if actions is not None:
            ha = self.a_encoder(actions.view(bsize, -1))
            z = self.z_network(torch.cat((h, ha), 1).view(bsize, -1))
            z_exp = self.z_expander(z)
            if save_z:
                self.save_z(z)
        else:
            z = self.sample_z(bsize)
            z_exp = self.z_expander(z)

        h = h + z_exp
        a = self.fc(h)
        a = a.view(bsize, self.opt.npred, self.opt.n_actions)
        return a, Variable(torch.zeros(1))


    def intype(self, t):
        if t == 'gpu':
            self.cuda()
            self.use_cuda = True
        elif t == 'cpu':
            self.cpu()
            self.use_cuda = False
