import torch
import torch.nn as nn
import random

from ppuu.modeling.common_models import Encoder, UNetwork, Decoder


class encoder(nn.Module):
    def __init__(
        self,
        opt,
        a_size,
        n_inputs,
        states=True,
        state_input_size=4,
        n_channels=3,
    ):
        super(encoder, self).__init__()
        self.opt = opt
        self.a_size = a_size
        self.n_inputs = opt.ncond if n_inputs is None else n_inputs
        self.n_channels = n_channels
        # frame encoder
        if opt.layers == 3:
            assert opt.nfeature % 4 == 0
            self.feature_maps = (
                opt.nfeature // 4,
                opt.nfeature // 2,
                opt.nfeature,
            )
            self.f_encoder = nn.Sequential(
                nn.Conv2d(
                    n_channels * self.n_inputs, self.feature_maps[0], 4, 2, 1
                ),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[0], self.feature_maps[1], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[1], self.feature_maps[2], 4, 2, 1),
            )
        elif opt.layers == 4:
            assert opt.nfeature % 8 == 0
            self.feature_maps = (
                opt.nfeature // 8,
                opt.nfeature // 4,
                opt.nfeature // 2,
                opt.nfeature,
            )
            self.f_encoder = nn.Sequential(
                nn.Conv2d(
                    n_channels * self.n_inputs, self.feature_maps[0], 4, 2, 1
                ),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[0], self.feature_maps[1], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[1], self.feature_maps[2], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[2], self.feature_maps[3], 4, 2, 1),
            )

        if states:
            n_hidden = self.feature_maps[-1]
            # state encoder
            self.s_encoder = nn.Sequential(
                nn.Linear(state_input_size * self.n_inputs, n_hidden),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(n_hidden, n_hidden),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(n_hidden, opt.hidden_size),
            )

        if a_size > 0:
            # action or cost encoder
            n_hidden = self.feature_maps[-1]
            self.a_encoder = nn.Sequential(
                nn.Linear(a_size, n_hidden),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(n_hidden, n_hidden),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(n_hidden, opt.hidden_size),
            )

    def forward(self, images, states=None, actions=None):
        bsize = images.size(0)
        h = self.f_encoder(
            images.view(
                bsize,
                self.n_inputs * self.n_channels,
                self.opt.height,
                self.opt.width,
            )
        )
        if states is not None:
            h = h + self.s_encoder(states.contiguous().view(bsize, -1)).view(
                h.size()
            )
        if actions is not None:
            a = self.a_encoder(actions.contiguous().view(bsize, self.a_size))
            h = h + a.view(h.size())
        return h


class u_network(nn.Module):
    def __init__(self, opt):
        super(u_network, self).__init__()
        self.opt = opt
        self.encoder = nn.Sequential(
            nn.Conv2d(self.opt.nfeature, self.opt.nfeature, 4, 2, 1),
            nn.Dropout2d(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.opt.nfeature, self.opt.nfeature, (4, 1), 2, 1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.opt.nfeature, self.opt.nfeature, (4, 1), 2, 1
            ),
            nn.Dropout2d(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                self.opt.nfeature, self.opt.nfeature, (4, 3), 2, 0
            ),
        )

        assert self.opt.layers == 3  # hardcoded sizes
        self.hidden_size = self.opt.nfeature * 3 * 2
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.opt.nfeature),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.hidden_size),
        )

    def forward(self, h):
        h1 = self.encoder(h)
        h2 = self.fc(h1.view(-1, self.hidden_size))
        h2 = h2.view(h1.size())
        h3 = self.decoder(h2)
        return h3


class decoder(nn.Module):
    def __init__(self, opt, n_out=1):
        super(decoder, self).__init__()
        self.opt = opt
        self.n_out = n_out
        if self.opt.layers == 3:
            assert opt.nfeature % 4 == 0
            self.feature_maps = [
                int(opt.nfeature / 4),
                int(opt.nfeature / 2),
                opt.nfeature,
            ]
            self.f_decoder = nn.Sequential(
                nn.ConvTranspose2d(
                    self.feature_maps[2], self.feature_maps[1], (4, 4), 2, 1
                ),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(
                    self.feature_maps[1],
                    self.feature_maps[0],
                    (5, 5),
                    2,
                    (0, 1),
                ),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(
                    self.feature_maps[0], self.n_out * 3, (2, 2), 2, (0, 1)
                ),
            )

            self.h_reducer = nn.Sequential(
                nn.Conv2d(self.feature_maps[2], self.feature_maps[2], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(
                    self.feature_maps[2],
                    self.feature_maps[2],
                    (4, 1),
                    (2, 1),
                    0,
                ),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
            )

        elif self.opt.layers == 4:
            assert opt.nfeature % 8 == 0
            self.feature_maps = [
                int(opt.nfeature / 8),
                int(opt.nfeature / 4),
                int(opt.nfeature / 2),
                opt.nfeature,
            ]

            self.f_decoder = nn.Sequential(
                nn.ConvTranspose2d(
                    self.feature_maps[3], self.feature_maps[2], (4, 4), 2, 1
                ),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(
                    self.feature_maps[2],
                    self.feature_maps[1],
                    (5, 5),
                    2,
                    (0, 1),
                ),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(
                    self.feature_maps[1],
                    self.feature_maps[0],
                    (2, 4),
                    2,
                    (1, 0),
                ),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(
                    self.feature_maps[0], self.n_out * 3, (2, 2), 2, (1, 0)
                ),
            )

            self.h_reducer = nn.Sequential(
                nn.Conv2d(opt.nfeature, opt.nfeature, (4, 1), (2, 1), 0),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
            )

        n_hidden = self.feature_maps[-1]

        self.s_predictor = nn.Sequential(
            nn.Linear(2 * n_hidden, n_hidden),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_hidden, self.n_out * 4),
        )

    def forward(self, h):
        bsize = h.size(0)
        h = h.view(
            bsize, self.feature_maps[-1], self.opt.h_height, self.opt.h_width
        )
        h_reduced = self.h_reducer(h).view(bsize, -1)
        pred_state = self.s_predictor(h_reduced)
        pred_image = self.f_decoder(h)
        pred_image = pred_image[
            :, :, : self.opt.height, : self.opt.width
        ].clone()
        pred_image = pred_image.view(
            bsize, 1, 3 * self.n_out, self.opt.height, self.opt.width
        )
        return pred_image, pred_state


###############
# Main models
###############

# forward model, deterministic (compatible with TEN3 model, use to initialize)
class FwdCNN(nn.Module):
    def __init__(self, opt, mfile):
        super(FwdCNN, self).__init__()
        self.opt = opt
        # If we are given a model file, use it to initialize this model.
        # otherwise initialize from scratch
        if mfile == "":
            self.encoder = encoder(opt, 0, opt.ncond)
            self.decoder = Decoder(
                layers=opt.layers,
                n_feature=opt.nfeature,
                dropout=opt.dropout,
                h_height=opt.h_height,
                h_width=opt.h_width,
                height=opt.height,
                width=opt.width,
            )
            self.a_encoder = nn.Sequential(
                nn.Linear(self.opt.n_actions, self.opt.nfeature),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.opt.nfeature, self.opt.nfeature),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.opt.nfeature, self.opt.hidden_size),
            )
            self.u_network = UNetwork(
                n_feature=opt.nfeature, layers=opt.layers, dropout=opt.dropout
            )
        else:
            print("[initializing encoder and decoder with: {}]".format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)["model"]
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.a_encoder = pretrained_model.a_encoder
            self.u_network = pretrained_model.u_network
            self.encoder.n_inputs = opt.ncond

    # dummy function
    def sample_z(self, bsize, method=None):
        return torch.zeros(bsize, 32).cuda()

    def forward_single_step(self, input_images, input_states, action, z):
        # encode the inputs (without the action)
        bsize = input_images.size(0)
        h_x = self.encoder(input_images, input_states)
        h_x = h_x.view(
            bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width
        )
        a_emb = self.a_encoder(action).view(h_x.size())

        h = h_x
        h = h + a_emb
        h = h + self.u_network(h)
        pred_image, pred_state = self.decoder(h)
        pred_image = torch.sigmoid(
            pred_image + input_images[:, -1].unsqueeze(1)
        )
        pred_state = pred_state + input_states[:, -1]
        return pred_image, pred_state

    def forward(self, inputs, actions, target, sampling=None, z_dropout=None):
        npred = actions.size(1)
        input_images, input_states = inputs
        pred_images, pred_states = [], []
        for t in range(npred):
            h = self.encoder(input_images, input_states)
            a_emb = self.a_encoder(actions[:, t]).view(h.size())
            h = h + a_emb
            h = h + self.u_network(h)
            pred_image, pred_state = self.decoder(h)
            pred_image = torch.sigmoid(
                pred_image + input_images[:, -1].unsqueeze(1)
            )
            pred_state = pred_state + input_states[:, -1]
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat(
                (input_states[:, 1:], pred_state.unsqueeze(1)), 1
            )
            pred_images.append(pred_image)
            pred_states.append(pred_state)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        return [pred_images, pred_states, None], torch.zeros(1).cuda()


# this version adds the actions *after* the z variables
class FwdCNN_VAE(nn.Module):
    def __init__(self, opt, mfile=""):
        super(FwdCNN_VAE, self).__init__()
        self.opt = opt

        if mfile == "":
            self.encoder = encoder(opt, 0, opt.ncond)
            self.decoder = Decoder(
                layers=opt.layers,
                n_feature=opt.nfeature,
                dropout=opt.dropout,
                h_height=opt.h_height,
                h_width=opt.h_width,
                height=opt.height,
                width=opt.width,
            )
            self.a_encoder = nn.Sequential(
                nn.Linear(self.opt.n_actions, self.opt.nfeature),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.opt.nfeature, self.opt.nfeature),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.opt.nfeature, self.opt.hidden_size),
            )
            self.u_network = UNetwork(
                n_feature=opt.nfeature, layers=opt.layers, dropout=opt.dropout
            )
        else:
            print("[initializing encoder and decoder with: {}]".format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)
            if type(pretrained_model) is dict:
                pretrained_model = pretrained_model["model"]
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.a_encoder = pretrained_model.a_encoder
            self.u_network = pretrained_model.u_network
            self.encoder.n_inputs = opt.ncond
            self.decoder.n_out = 1

        self.y_encoder = Encoder(
            Encoder.Config(a_size=0, n_inputs=1, states=False)
        )

        self.z_network = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.nfeature),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, 2 * opt.nz),
        )

        self.z_zero = torch.zeros(self.opt.batch_size, self.opt.nz)
        self.z_expander = nn.Linear(opt.nz, opt.hidden_size)

    def reparameterize(self, mu, logvar, sample):
        if self.training or sample:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def sample_z(self, bsize, method=None, h_x=None):
        z = torch.randn(bsize, self.opt.nz).cuda()
        return z

    def forward_single_step(self, input_images, input_states, action, z):
        # encode the inputs (without the action)
        bsize = input_images.size(0)
        h_x = self.encoder(input_images, input_states)
        z_exp = self.z_expander(z).view(
            bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width
        )
        h_x = h_x.view(
            bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width
        )
        a_emb = self.a_encoder(action).view(h_x.size())

        h = h_x + z_exp
        h = h + a_emb
        h = h + self.u_network(h)
        pred_image, pred_state = self.decoder(h)
        pred_image = torch.sigmoid(
            pred_image + input_images[:, -1].unsqueeze(1)
        )
        pred_state = pred_state + input_states[:, -1]

        return pred_image, pred_state

    def forward(
        self,
        inputs,
        actions,
        targets,
        save_z=False,
        sampling=None,
        z_dropout=0.0,
        z_seq=None,
        noise=None,
    ):
        input_images, input_states = inputs
        bsize = input_images.size(0)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)
        ploss = torch.zeros(1).cuda()
        ploss2 = torch.zeros(1).cuda()

        pred_images, pred_states = [], []
        z_list = []

        z = None
        for t in range(npred):
            # encode the inputs (without the action)
            h_x = self.encoder(input_images, input_states)
            if sampling is None:
                # we are training or estimating z distribution
                target_images, target_states, _ = targets
                # encode the targets into z
                h_y = self.y_encoder(
                    target_images[:, t].unsqueeze(1).contiguous()
                )
                if random.random() < z_dropout:
                    z = self.sample_z(bsize, method=None, h_x=h_x).data
                else:
                    mu_logvar = self.z_network(
                        (h_x + h_y).view(bsize, -1)
                    ).view(bsize, 2, self.opt.nz)
                    mu = mu_logvar[:, 0]
                    logvar = mu_logvar[:, 1]
                    z = self.reparameterize(mu, logvar, True)
                    # this can go to inf when taking exp(), so clamp it
                    logvar = torch.clamp(logvar, max=4)
                    if self.opt.model == "fwd-cnn-vae-fp":
                        kld = -0.5 * torch.sum(
                            1 + logvar - mu.pow(2) - logvar.exp()
                        )
                        kld /= bsize
                        ploss += kld
                    else:
                        raise ValueError
            else:
                if z_seq is not None:
                    z = z_seq[t]
                else:
                    z = self.sample_z(bsize, method=None, h_x=h_x)

            z_list.append(z)
            z_exp = self.z_expander(z).view(
                bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width
            )
            h_x = h_x.view(
                bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width
            )
            h = h_x + z_exp
            a_emb = self.a_encoder(actions[:, t]).view(h.size())
            h = h + a_emb
            h = h + self.u_network(h)

            pred_image, pred_state = self.decoder(h)
            if sampling is not None:
                pred_image.detach()
                pred_state.detach()
            pred_image = torch.sigmoid(
                pred_image + input_images[:, -1].unsqueeze(1)
            )
            pred_state = pred_state + input_states[:, -1]

            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat(
                (input_states[:, 1:], pred_state.unsqueeze(1)), 1
            )
            pred_images.append(pred_image)
            pred_states.append(pred_state)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        z_list = torch.stack(z_list, 1)
        return [pred_images, pred_states, z_list], [ploss, ploss2]

    def intype(self, t):
        if t == "gpu":
            self.cuda()
            self.z_zero = self.z_zero.cuda()
            self.use_cuda = True
        elif t == "cpu":
            self.cpu()
            self.use_cuda = False
            self.z_zero = self.z_zero.cpu()
